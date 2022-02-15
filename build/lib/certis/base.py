import pandas as pd
import os, json
from typing import *


class Action:
    """
    Abstract class for Order, OrderCancellation
    """
    def __init__(self):
        pass


class Strategy:
    """
    Abstract method for generating user-defined trading strategies with certis
    """
    def __init__(self, config, name="CertisStrategy"):
        self.config = config
        self.name = name

    def _calculate(self, data):
        return data

    def calculate(self, data: pd.DataFrame):
        return self._calculate(data)

    def execute(self, state_dict: Dict[str, Any]) -> List[Action]:
        raise NotImplementedError


class Logger:
    """
    Logger Object
    """
    def __init__(self):
        self._transactions = []
        self._account_infos = []
        self._unfilled_orders = []

    @property
    def transactions(self) -> List[Dict[str, Any]]:
        """
        list of transactions during the backtest.
        each transaction is generated when order fills

        :return: self._transactions
        """
        return self._transactions

    @property
    def account_infos(self) -> List[Dict[str, Any]]:
        """
        account infos during the backtest.
        recorded interval-by-interval

        :return: self._account_infos
        """
        return self._account_infos

    @property
    def unfilled_orders(self) -> List[Dict[str, Any]]:
        """
        unfilled orders during the backtest.
        recorded interval-by-interval

        :return: self._unfilled_orders
        """
        return self._unfilled_orders

    def add_transaction(self, transactions: List[Dict[str, Any]]) -> None:
        """
        adds transactions

        :param transactions: transactions
        :return: None
        """
        self._transactions.extend(transactions)

    def add_account_info(self, account_info: Dict[str, Any]) -> None:
        """
        adds account info

        :param account_info: account info
        :return: None
        """
        self._account_infos.append(account_info)

    def add_unfilled_orders(self, unfilled_orders: Dict[str, Any]) -> None:
        """
        adds unfilled orders

        :param unfilled_orders: unfilled orders
        :return: None
        """
        self._unfilled_orders.append(unfilled_orders)

    def to_json(self, target_directory: str) -> None:
        """
        writes logger to json

        :param target_directory: target directory to write
        :return: None
        """
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        with open(os.path.join(target_directory, "transactions.json"), "w") as f:
            f.write(json.dumps(self._transactions))

        with open(os.path.join(target_directory, "account_infos.json"), "w") as f:
            f.write(json.dumps(self._account_infos))

        with open(os.path.join(target_directory, "unfilled_orders.json"), "w") as f:
            f.write(json.dumps(self._unfilled_orders))


