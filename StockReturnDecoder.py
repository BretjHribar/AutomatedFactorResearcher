import numpy as np

class StockReturnDecoder:
    """
    Decodes stock market expected returns from alpha expected returns using
    position data, as described in "Decoding Stock Market with Quant Alphas"
    by Zura Kakushadze and Willie Yu.

    This implementation includes handling of linear constraints and allows for
    modeling the covariance matrix of regression residuals using principal
    components.
    """

    def __init__(self):
        pass

    def _calculate_effective_rank(self, x, exclude_first=True):
        """
        Calculates the effective rank of a matrix using the entropy-based method.

        Args:
            x (np.ndarray): Input matrix.
            exclude_first (bool): Whether to exclude the largest eigenvalue.

        Returns:
            float: Effective rank.
        """
        positive_eigenvalues = x[x > 0]
        if exclude_first:
            positive_eigenvalues = positive_eigenvalues[1:]
        probabilities = positive_eigenvalues / np.sum(positive_eigenvalues)
        entropy = -np.sum(probabilities * np.log(probabilities))
        effective_rank = np.exp(entropy)
        if exclude_first:
            effective_rank += 1
        return effective_rank

    def _calculate_specific_variance(self, residuals, num_components, truncate=True):
        """
        Calculates the specific variance of regression residuals, optionally
        modeling the covariance matrix using principal components.

        Args:
            residuals (np.ndarray): Matrix of regression residuals (N x (d-1)).
            num_components (int): Number of principal components to use (K).
                If 0 or >= d-2, no PCA is performed. If < 0, eRank is used.
            truncate (bool): Whether to truncate eRank to an integer.

        Returns:
            np.ndarray: Vector of specific variances (N,).
        """
        num_alphas, time_steps = residuals.shape
        if num_components == 0 or num_components >= time_steps - 1:
            # No PCA, specific variance is just the variance of each row
            return np.var(residuals, axis=1)

        # Calculate eigenvalues and eigenvectors of the covariance matrix
        # Dimensions: eigenvalues (d-1,), eigenvectors ((d-1) x (d-1))
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(residuals.T))

        if num_components < 0:
            # Use effective rank to determine the number of components
            effective_rank = self._calculate_effective_rank(eigenvalues, exclude_first=False)
            num_components = int(effective_rank) if truncate else round(effective_rank)

        # Select the top K principal components
        selected_eigenvalues = eigenvalues[-num_components:]  # (K,)
        selected_eigenvectors = eigenvectors[:, -num_components:]  # ((d-1) x K)

        # Calculate specific variance using the selected components
        # Dimensions: ((d-1) x K) * (K,) * (K x (d-1)) -> (d-1) x (d-1) -> (d-1,)
        return np.sum((selected_eigenvectors**2 * selected_eigenvalues).T, axis=1)

    def decode_stock_returns(self, alpha_expected_returns, stock_positions, num_components, tolerance=1e-8):
        """
        Extracts stock expected returns from alpha expected returns.

        Args:
            alpha_expected_returns (np.ndarray): Matrix of alpha expected returns (N x d).
            stock_positions (np.ndarray): 3D array of stock positions (N x M x d).
            num_components (int): Number of principal components to use for modeling
                the covariance matrix of regression residuals.
            tolerance (float): Tolerance for identifying null eigenvalues.

        Returns:
            np.ndarray: Vector of stock expected returns (M,).
        """
        num_alphas, time_steps = alpha_expected_returns.shape
        num_stocks = stock_positions.shape[1]

        # Calculate regression residuals for each time step
        residuals = np.zeros((num_alphas, time_steps - 1))  # (N x (d-1))
        for s in range(1, time_steps):
            # Dimensions: (N,) = (N,) - (N x M) * (M,)
            residuals[:, s - 1] = alpha_expected_returns[:, s] - stock_positions[:, :, s] @ np.linalg.lstsq(
                stock_positions[:, :, s], alpha_expected_returns[:, s], rcond=None
            )[0]

        # Calculate regression weights (inverse of specific variances)
        # Dimensions: (N,)
        regression_weights = 1 / self._calculate_specific_variance(residuals, num_components)

        # Normalize regression weights to sum to 1
        ###regression_weights /= np.sum(regression_weights)

        # Calculate X matrix (weighted sum of outer products of positions)
        # Dimensions: (M x M) = (M x N) * (N x N) * (N x M)
        # X = (
        #     stock_positions[:, :, 0].T
        #     @ (np.diag(regression_weights) @ stock_positions[:, :, 0])
        # )

        X = stock_positions[:, :, 0].T @ stock_positions[:, :, 0]

        # Calculate eigenvalues and eigenvectors of X
        # Dimensions: eigenvalues (M,), eigenvectors (M x M)
        eigenvalues, eigenvectors = np.linalg.eigh(X)

        # Select non-null eigenvalues and corresponding eigenvectors
        non_null_indices = eigenvalues > tolerance * eigenvalues[0]
        non_null_eigenvalues = eigenvalues[non_null_indices]  # (M-p,)
        non_null_eigenvectors = eigenvectors[:, non_null_indices]  # (M x (M-p))

        # Calculate stock expected returns
        # Dimensions: (M,) = (M x (M-p)) * ((M-p) x (M-p)) * ((M-p) x 1)
        stock_returns = (
            non_null_eigenvectors
            @ np.diag(1 / non_null_eigenvalues)
            @ non_null_eigenvectors.T
            @ ((stock_positions[:, :, 0] * regression_weights[:, None]).T @ alpha_expected_returns[:, 0])
        )


        return stock_returns