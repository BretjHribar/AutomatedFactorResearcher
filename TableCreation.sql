#ALPHAS
CREATE TABLE `alphas` (
  `alphasid` int NOT NULL AUTO_INCREMENT,
  `strategy_id` int DEFAULT NULL,
  `universe` varchar(100) DEFAULT NULL,
  `scriptversion` varchar(100) DEFAULT NULL,
  `alphastring` varchar(2000) DEFAULT NULL,
  `sharpe` float DEFAULT NULL,
  `fitness` float DEFAULT NULL,
  `turnover` float DEFAULT NULL,
  `returns` float DEFAULT NULL,
  `margin` float DEFAULT NULL,
  `topN` int DEFAULT NULL,
  `PSR` float DEFAULT NULL,
  `riskModelType` varchar(20) DEFAULT NULL,
  `trialCounter` int DEFAULT NULL,
  `feesBSP` float DEFAULT NULL,
  `universeBlocking` tinyint DEFAULT NULL,
  `targetDelay` tinyint DEFAULT NULL,
  `hedgeVol` tinyint DEFAULT NULL,
  `rankHedge` tinyint DEFAULT NULL,
  `hedgeIndustry` tinyint DEFAULT NULL,
  `corrCutoff` float DEFAULT NULL,
  `date` datetime DEFAULT NULL,
  `corr` float DEFAULT NULL,
  `lineardecay` int DEFAULT NULL,
  `DSRprob` float DEFAULT NULL,
  `ISOSratio` float DEFAULT NULL,
  `minPrice` float DEFAULT NULL,
  `maxPrice` float DEFAULT NULL,
  `maxStockWeight` float DEFAULT NULL,
  PRIMARY KEY (`alphasid`)
) ENGINE=InnoDB AUTO_INCREMENT=3491150 DEFAULT CHARSET=utf8mb3;

#STRATEGIES
CREATE TABLE `strategies` (
  `strategy_id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(45) DEFAULT NULL,
  `universe` varchar(45) DEFAULT NULL,
  `riskModelType` varchar(45) DEFAULT NULL,
  `featuresList` varchar(2000) DEFAULT NULL,
  `GPfunctionsList` varchar(2000) DEFAULT NULL,
  PRIMARY KEY (`strategy_id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
