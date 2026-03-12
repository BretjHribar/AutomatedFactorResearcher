"""Background redemption daemon — runs every 60s, redeems ALL positions."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()
from web3 import Web3
from polymarket_api import PolymarketClient

TRADE_LOG = "real_trades.jsonl"
w3 = Web3(Web3.HTTPProvider('https://polygon-bor-rpc.publicnode.com', request_kwargs={'timeout': 15}))
pk = os.getenv('POLYGON_PRIVATE_KEY')
acct = w3.eth.account.from_key(pk)

CT = Web3.to_checksum_address('0x4D97DCd97eC945f40cF65F87097ACe5EA0476045')
USDC_ADDR = Web3.to_checksum_address('0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174')

CT_ABI = [
    {'constant': True, 'inputs': [{'name': 'account', 'type': 'address'},{'name': 'id', 'type': 'uint256'}],
     'name': 'balanceOf', 'outputs': [{'name': '', 'type': 'uint256'}], 'type': 'function'},
    {'constant': False, 'inputs': [
        {'name': 'collateralToken', 'type': 'address'},{'name': 'parentCollectionId', 'type': 'bytes32'},
        {'name': 'conditionId', 'type': 'bytes32'},{'name': 'indexSets', 'type': 'uint256[]'}
    ], 'name': 'redeemPositions', 'outputs': [], 'type': 'function'},
    {'constant': True, 'inputs': [{'name': 'conditionId', 'type': 'bytes32'}],
     'name': 'payoutDenominator', 'outputs': [{'name': '', 'type': 'uint256'}], 'type': 'function'},
]
USDC_ABI = [{'constant': True, 'inputs': [{'name': '_owner', 'type': 'address'}],
     'name': 'balanceOf', 'outputs': [{'name': 'balance', 'type': 'uint256'}], 'type': 'function'}]

ct = w3.eth.contract(address=CT, abi=CT_ABI)
usdc = w3.eth.contract(address=USDC_ADDR, abi=USDC_ABI)

redeemed = set()

def sweep():
    conditions = {}
    if not os.path.exists(TRADE_LOG):
        return
    with open(TRADE_LOG) as f:
        for line in f:
            t = json.loads(line)
            cond = t.get('condition_id', '')
            token = t.get('full_token_id', '')
            slug = t.get('slug', '')
            if cond and token and cond not in redeemed:
                conditions[cond] = {'token': token, 'slug': slug}

    if not conditions:
        return

    nonce = w3.eth.get_transaction_count(acct.address)
    gas_price = w3.eth.gas_price

    for cond_hex, info in conditions.items():
        try:
            token_id = int(info['token'])
            bal = ct.functions.balanceOf(acct.address, token_id).call()
            if bal == 0:
                redeemed.add(cond_hex)
                continue

            cond_bytes = Web3.to_bytes(hexstr=cond_hex)
            payout = ct.functions.payoutDenominator(cond_bytes).call()
            if payout == 0:
                continue  # Not settled yet

            usdc_pre = usdc.functions.balanceOf(acct.address).call()
            tx = ct.functions.redeemPositions(
                USDC_ADDR, bytes(32), cond_bytes, [1, 2]
            ).build_transaction({
                'from': acct.address, 'nonce': nonce,
                'gasPrice': gas_price, 'gas': 200000
            })
            signed = w3.eth.account.sign_transaction(tx, pk)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)
            usdc_post = usdc.functions.balanceOf(acct.address).call()
            gained = (usdc_post - usdc_pre) / 1e6
            redeemed.add(cond_hex)
            nonce += 1
            ts = time.strftime("%H:%M:%S")
            print(f"  {ts} REDEEM {info['slug']}: +${gained:.4f}", flush=True)
        except Exception as e:
            if 'nonce' in str(e).lower():
                nonce = w3.eth.get_transaction_count(acct.address)
            continue


if __name__ == "__main__":
    print(f"Redemption daemon started. Wallet: {acct.address}", flush=True)
    bal = usdc.functions.balanceOf(acct.address).call() / 1e6
    print(f"USDC: ${bal:.2f}", flush=True)
    print("Sweeping every 60 seconds...", flush=True)
    
    while True:
        try:
            sweep()
        except Exception as e:
            print(f"  Sweep error: {e}", flush=True)
        time.sleep(30)  # Sweep every 30s for faster capital recovery
