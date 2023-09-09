import random
import pymysql.cursors
from Constants import MASTER_FEATURES_LIST, MASTER_GPFUNCTIONS_LIST
import Constants

strategyName = "TEST_STRATEGY_9"
fractionOfFeatures = 0.5
fractionOfFunctions = 0.5
riskModelType = Constants.SUB_INDUSTRY_RISK_MODEL

def InsertNewStrategy(strategyName, universe, features, functionsList):
    try:
        with connection.cursor() as cursor:
            sql = "INSERT INTO `quantschema`.`strategies` " \
                  "(`name`,`universe`,`featuresList`,`GPfunctionsList`) " \
                  "VALUES (%s,%s,%s,%s)"
            # if the connection was lost, then it reconnects
            connection.ping(reconnect=True)
            cursor.execute(sql, (
            str(strategyName), str(universe), str(features), str(functionsList)))
            connection.commit()
            print("inserted new strategy into DB: ")
    finally:
        pass
    return 1

connection = pymysql.connect(host='alphasdatabase1.cysvmgsjf7ox.us-east-1.rds.amazonaws.com',#'localhost',
                             user='admin', #mysqluser',
                             password='SALMON44', #'mysqluser',
                             db='quantschema',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

strategyFeatures = random.sample(MASTER_FEATURES_LIST, k=round(len(MASTER_FEATURES_LIST) * fractionOfFeatures))
print(strategyFeatures)
strategyFunctions = random.sample(MASTER_GPFUNCTIONS_LIST, k=round(len(MASTER_GPFUNCTIONS_LIST) * fractionOfFunctions))
print(strategyFunctions)

InsertNewStrategy(strategyName, "YFINANCE", strategyFeatures, strategyFunctions)

