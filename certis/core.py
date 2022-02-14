from certis.util import *
from certis.base import *
from certis.constants import *
from typing import *

import tqdm
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class MarketInfo:
    """
    takes & contains all these market information we need
    """

    def __init__(
            self,
            maker_fee: float,
            taker_fee: float,
            slippage: float,
            tick_size: float,
            minimum_order_size: float,
            **kwargs,
    ):
        """
        initializes MarketInfo class, takes all these market information we need

        :param maker_fee:
        maker fee, fee for market orders. 1% = 0.01
        :param taker_fee:
        taker fee, fee for limit orders. 1% = 0.01
        :param slippage:
        slippage for market orders. 1% = 0.01
        :param tick_size:
        tick size for this data. in other words, minimum change unit.
        like (123.123, 12.124, 12.122 ... ), tick size is 0.001
        :param minimum_order_size:
        minimum order size for this data.
        """

        self._maker_fee = maker_fee
        self._taker_fee = taker_fee
        self._slippage = slippage
        self._tick_size = tick_size
        self._minimum_order_size = minimum_order_size

    @property
    def maker_fee(self):
        """
        :return: maker fee, fee for market orders. 1% = 0.01
        """
        return self._maker_fee

    @property
    def taker_fee(self):
        """
        :return: taker fee, fee for limit orders. 1% = 0.01
        """
        return self._taker_fee

    @property
    def slippage(self):
        """
        slippage for market orders. 1% = 0.01

        :return: self._slippage
        """
        return self._slippage

    @property
    def tick_size(self):
        """
        tick size for this data.
        in other words, minimum change unit.
        like (123.123, 12.124, 12.122 ... ), tick size is 0.001

        :return: self._tick_size
        """
        return self._tick_size

    @property
    def minimum_order_size(self):
        """
        minimum order size for this data.

        :return: self._minimum_order_size
        """
        return self._minimum_order_size

    def trim_order_size(self, size: Optional[float]) -> Optional[float]:
        """
        trims order size by doing
        (size // minimum order size) * minimum order size

        :param size:  order's quantity
        :return: order size, trimmed by minimum order size
        """
        if size is None:
            return None

        return (size // self._minimum_order_size) * self._minimum_order_size

    def trim_order_price(self, price: float) -> float:
        """
        trims order price by doing
        (price // tick size) * tick size

        :param price: ordered price
        :return: trimmed order price
        """
        return (
            (price // self._tick_size) * self._tick_size
            if price is not None
            else None
        )

    def apply_slippage(self, price: float, side: int) -> float:
        """
        applies slippage for given price and side
        for side: long -> higher price
        for side: short -> lower price

        :param price: order price
        :param side: order side
        :return: slippage-applied order price
        """
        return (
            self.trim_order_price(price * (1 + side * self._slippage))
            if price is not None
            else None
        )


class Order(Action):
    """
    Order object in Certis
    """

    def __init__(
            self,
            order_side=None,
            order_quantity=None,
            order_type: str = None,
            order_price: Optional[np.float64] = None,
            reduce_only: bool = False,
    ):
        super(Order, self).__init__()
        self._id = generate_random_string()

        self._side = order_side
        self._quantity = order_quantity

        self._type = order_type
        self._price = order_price
        self._reduce_only = reduce_only

        if self._type == OrderType.MARKET:
            self._price = None

        if self._type in (
                OrderType.STOP_LOSS_MARKET,
                OrderType.TAKE_PROFIT_MARKET,
        ):
            self._reduce_only = True

        self._check_validity()

    def __dict__(self) -> Dict[str, Any]:
        """
        converts order object as dict
        for logging

        :return: order object as dict
        """
        return {
            "id": self._id,
            "side": self._side,
            "quantity": self._quantity,
            "reduce_only": int(self._reduce_only),
            "type": self._type,
            "price": self._price,
        }

    def __str__(self):
        """
        converts order object to string
        for logging

        :return: order object as string
        """
        return f"""
        Order:
            id : {self._id},
            side : {self._side},
            quantity : {self._quantity},
            reduce_only : {self._reduce_only},
            type : {self._type},
            price : {self._price}
            """

    @property
    def id(self) -> str:
        """
        order's id

        :return: self._id
        """
        return self._id

    @property
    def quantity(self) -> float:
        """
        order's quantity

        :return: self._quantity
        """
        return self._quantity

    @property
    def side(self) -> int:
        """
        order side defined in certis.core.OrderSide
        required in every orders except for STOP_LOSS_MARKET & TAKE_PROFIT_MARKET orders

        :return: self._side
        """
        return self._side

    @property
    def price(self) -> float:
        """
        order's price
        required in LIMIT, STOP_MARKET (stop price) orders

        :return: self._price
        """
        return self._price

    @property
    def type(self) -> str:
        """
        order's type
        defined in certis.constants.OrderType

        :return:
        """
        return self._type

    @property
    def reduce_only(self) -> bool:
        """
        if order is reduce-only order or not

        :return: self._reduce_only
        """
        return self._reduce_only

    def _check_validity(self) -> None:
        """
        checks order's validity
        raises ValueError if order is invalid

        :return: None
        """
        self._check_order_side_validity()
        self._check_order_type_validity()

    def _check_order_side_validity(self) -> None:
        """
        checks order side validity
        order except type=SL/TP should have one side, LONG or SHORT

        :return: None
        """
        if not self._type in [OrderType.STOP_LOSS_MARKET, OrderType.TAKE_PROFIT_MARKET]:
            if not self._side in OrderSide.SIDES:
                raise ValueError("got invalid order side: {}".format(self._side))

    def _check_order_type_validity(self) -> None:
        """
        checks order type validity
        order type should be in OrderType.ORDERS
        order except type=SL/TP should have one side, LONG or SHORT
        non-market orders should have order price

        :return: None
        """

        if not self._type in OrderType.ORDERS:
            raise ValueError("Got Invalid Order: {}".format(self._type))

        if not self._type in [OrderType.STOP_LOSS_MARKET, OrderType.TAKE_PROFIT_MARKET]:
            if self.quantity is None or self.side is None:
                raise ValueError("quantity and side is nesscery except TP/SL Orders")

        if (
                self._type
                in [
                    OrderType.LIMIT,
                    OrderType.STOP_MARKET,
                    OrderType.STOP_LOSS_MARKET,
                    OrderType.TAKE_PROFIT_MARKET,
                ]
        ) & (self._price is None):
            raise ValueError(
                "When Comes to non-Market Orders (in this case, {}), you have to set order_price".format(
                    self._type
                )
            )

    def check_order_price_validity(self, market_price: float) -> None:
        """
        checks order price's validity
        for certain cases that could raise "Order Could Execute Immediately" Error in live trading.

        :param market_price: market price (newest close price in this case)
        :return: None
        """
        if self._type == OrderType.LIMIT:
            if (self._price > market_price) & (self._side == OrderSide.LONG):
                raise ValueError("LIMIT ORDER ERROR")
            elif (self._price < market_price) & (
                    self._side == OrderSide.SHORT
            ):
                raise ValueError(
                    "LIMIT ORDER ERROR: SIDE=SHORT BUT PRICE < MARKET_PRICE"
                )
        if self._type == OrderType.STOP_MARKET:
            if (self._price < market_price) & (self._side == OrderSide.LONG):
                raise ValueError(
                    "STOP_MARKET ORDER ERROR: SIDE=LONG BUT PRICE < MARKET_PRICE"
                )
            elif (self._price > market_price) & (
                    self._side == OrderSide.SHORT
            ):
                raise ValueError(
                    "STOP_MARKET ORDER ERROR: SIDE=SHORT BUT PRICE > MARKET_PRICE"
                )

        if self._type == OrderType.STOP_LOSS_MARKET:
            if (self._price > market_price) & (self._side == OrderSide.LONG):
                raise ValueError(
                    "STOP_LOSS_MARKET ORDER ERROR: SIDE=LONG BUT PRICE > MARKET_PRICE"
                )
            elif (self._price < market_price) & (self._side == OrderSide.SHORT):
                raise ValueError(
                    "STOP_LOSS_MARKET ORDER ERROR: SIDE=SHORT BUT PRICE < MARKET_PRICE"
                )

        if self._type == OrderType.TAKE_PROFIT_MARKET:
            if (self._price < market_price) & (self._side == OrderSide.LONG):
                raise ValueError(
                    "TAKE_PROFIT_MARKET ORDER ERROR: SIDE=LONG BUT PRICE > MARKET_PRICE"
                )
            elif (self._price > market_price) & (self._side == OrderSide.SHORT):
                raise ValueError(
                    "TAKE_PROFIT_MARKET ORDER ERROR: SIDE=SHORT BUT PRICE < MARKET_PRICE"
                )

    def trim(self, market_info: MarketInfo) -> Action:
        """
        trims order itself

        :param market_info: market info Object for this backtest
        :return: self
        """
        self._price, self._quantity = (
            market_info.trim_order_price(self._price),
            market_info.trim_order_size(self._quantity),
        )
        return self

    def is_fillable_at(
            self,
            account_info: Dict[str, Any],
            market_info: MarketInfo,
            open_price: float,
            high_price: float,
            low_price: float,
    ) -> bool:
        if self._type == OrderType.MARKET:
            self._price = market_info.apply_slippage(open_price, self._side)
            return True

        elif self._type == OrderType.LIMIT:
            if self._side == OrderSide.SHORT:
                if self._price < high_price:
                    return True
                return False
            else:
                if self._price > low_price:
                    return True
                return False

        elif self._type == OrderType.STOP_MARKET:
            if (low_price < self._price) & (self._price < high_price):
                self._price = market_info.apply_slippage(
                    self._price, self._side
                )
                return True
            return False

        elif self._type == OrderType.STOP_LOSS_MARKET:
            if account_info["position"]["side"] == OrderSide.LONG:
                if self._price > low_price:
                    self._quantity = account_info["position"]["size"]
                    self._side = -account_info["position"]["side"]
                    self._price = market_info.apply_slippage(
                        self._price, self._side
                    )
                    return True
                return False

            if account_info["position"]["side"] == OrderSide.SHORT:
                if self._price < high_price:
                    self._quantity = account_info["position"]["size"]
                    self._side = -account_info["position"]["side"]
                    self._price = market_info.apply_slippage(
                        self._price, self._side
                    )
                    return True
                return False

        elif self._type == OrderType.TAKE_PROFIT_MARKET:
            if account_info["position"]["side"] == OrderSide.LONG:
                if self._price < high_price:
                    self._price = market_info.apply_slippage(
                        self._price, self._side
                    )
                    self._quantity = account_info["position"]["size"]
                    self._side = -account_info["position"]["side"]
                    return True
                return False

            if account_info["position"]["side"] == OrderSide.SHORT:
                if self._price > low_price:
                    self._quantity = account_info["position"]["size"]
                    self._side = -account_info["position"]["side"]
                    return True
                return False

        else:
            raise ValueError(f"Invalid Order Type: {self._type}")

    def set_id(self, id: int):
        self._id = id


class OrderCancellation(Action):
    """
    order cancellation object
    """

    def __init__(self, id):
        super(OrderCancellation, self).__init__()
        self._id = id

    @property
    def id(self) -> str:
        """
        id for order to cancel
        if id == "all": cancels all order

        :return:
        """
        return self._id

    def __str__(self) -> str:
        """
        :return: order cancellation object as string
        """
        return "OrderCancellation(id={})".format(self._id)


class Position:
    def __init__(self):
        self._initialize()

    def _initialize(self) -> None:
        """
        initializes position

        :return: None
        """
        self._size = 0
        self._side = 0
        self._avg_price = 0
        self._unrealized_pnl = 0

    @property
    def info(self) -> Dict[str, Any]:
        """
        position as dict object

        :return:
        """
        return {
            "size": self._size,
            "side": self._side,
            "avg_price": self._avg_price,
            "unrealized_pnl": self._unrealized_pnl,
        }

    @property
    def avg_price(self) -> float:
        """
        average entry price for this position

        :return: self._avg_price
        """
        return self._avg_price

    def update_unrealized_pnl(self, price: float) -> None:
        """
        updates unrealized pnl

        :param price: current price
        :return: None
        """
        self._unrealized_pnl = (
                (price - self.avg_price) * self._side * self._size
        )

    def _initialize_if_invalid_size(self, market_info: MarketInfo) -> None:
        """
        initializes if invalid size
        invalid size: size < minimum order size
        this is often caused because of the floating point bug
        this can be critical for backtesting
        so we take this as an exception

        :param market_info: MarketInfo object
        :return: None
        """
        if self._size < market_info.minimum_order_size:
            # 부동소수점 버그 처리
            self._initialize()

    def update(self, price: float, size: float, side: int, market_info: MarketInfo) -> float:
        """
        updates position with new transaction

        :param price: price of transaction
        :param size: quantity of transaction
        :param side: side of transaction
        :param market_info: MarketInfo object
        :return: realized profit and loss (p&L)
        """
        realized_pnl = 0

        if size == 0:
            return 0.

        if (self._side == side) | (self._side == 0):
            self._avg_price = (size * price + self._size * self._avg_price) / (
                    size + self._size
            )

        else:
            if self._size <= size:
                realized_pnl = (
                        (price - self._avg_price) * self._side * self._size
                )
            else:
                realized_pnl = (price - self._avg_price) * self._side * size

        new_position = size * side + self._size * self._side

        self._size, self._side = np.abs(new_position), np.sign(new_position)

        if not self._side:
            self._avg_price = 0

        self._initialize_if_invalid_size(market_info)

        return realized_pnl


class Account:
    """
    Certis Account Object
    """

    def __init__(self, margin: float, market_info: MarketInfo):
        self._margin = margin
        self._portfolio_value = margin
        self._position = Position()
        self._market_info: MarketInfo = market_info

    def update_portfolio_value(self, price: float) -> object:
        """
        updates portfolio value
        updates unrealized pnl

        :param price: current price
        :return: self
        """
        self._position.update_unrealized_pnl(price)
        self._portfolio_value = (
                self._position.info["unrealized_pnl"] + self._margin
        )

        return self

    def update_position(self, price: float, size: float, side: int):
        """
        updates position with new transaction

        :param price: price of transaction
        :param size: quantity of transaction
        :param side: side of transaction
        :param market_info: MarketInfo object
        :return: realized profit and loss (p&L)
        """

        ret = self._position.update(price, size, side, market_info=self._market_info)
        return ret

    @property
    def info(self) -> Dict[str, Any]:
        """
        gives position info as dictionary

        :return: position info as dictionary
        """
        position_info = self._position.info
        return {
            "margin": self._margin,
            "portfolio_value": self._portfolio_value,
            "position": position_info,
            "has_position": int(position_info["size"] >= self._market_info.minimum_order_size),
        }

    @property
    def margin(self) -> float:
        """
        current margin left

        :return: self._margin
        """
        return self._margin

    @property
    def position(self) -> Position:
        """
        current position object

        :return: self._position
        """
        return self._position

    def deposit(self, size: float) -> object:
        self._margin += size
        return self


class Broker:
    """
    Virtual Broker object for Certis
    """

    def __init__(self, market_info: MarketInfo, initial_margin: float):
        self._account: Account = Account(initial_margin, market_info)
        self._market_info: MarketInfo = market_info
        self._order_queue: Dict[str, Order] = dict()

    @property
    def account_info(self):
        """
        account information

        :return: self._account.info
        """
        return self._account.info

    def apply_actions(self, actions: List[Action], price: float) -> None:
        """
        applies actions,
        which is List of Order / OrderCancellation Objects

        :param actions: list of actions (Order / OrderCancellation Objects)
        :param price: current price
        :return: None
        """
        for action in actions:
            if isinstance(action, Order):
                action.check_order_price_validity(price)
                self._place_order(action)
            if isinstance(action, OrderCancellation):
                self._cancel_order(action)

    def _cancel_order(self, action: OrderCancellation) -> object:
        """
        executes OrderCancellation Object
        if OrderCancellation.id is all: cancels all orders

        :param action: OrderCancellation Object
        :return: self
        """
        if action.id.lower() == "all":
            self._order_queue = {}
            return
        del self._order_queue[action.id]

        return self

    def _place_order(self, order: Order) -> object:
        """
        places order in order_queue

        :param order: Order object
        :return: self
        """
        order.trim(self._market_info)

        if order.quantity == 0:
            return

        self._order_queue[order.id] = order

        return self

    def fill_pending_orders(
            self, timestamp: int, open_price: float, high_price: float, low_price: float
    ) -> object:
        """
        executes orders in order queue

        :param timestamp: current timestamp
        :param open_price: current open price
        :param high_price: current high price
        :param low_price: current low price
        :return: self
        """
        transactions = []

        for order_id in list(self._order_queue.keys()):
            order: Order = self._order_queue[order_id]

            if order._reduce_only & (
                    (not self._account.info["has_position"])
                    | (self._account.info["position"]["side"] == order._side)
            ):
                del self._order_queue[order.id]
                continue

            elif (order.quantity is not None) & (order.reduce_only):
                if order.quantity > self._account.info["position"]["size"]:
                    del self._order_queue[order.id]
                    continue

            if order.is_fillable_at(
                    self._account.info,
                    self._market_info,
                    open_price,
                    high_price,
                    low_price,
            ):
                realized_pnl = self._account.update_position(
                    order.price, order.quantity, order.side
                )

                order_amount = order.quantity * order.price

                fee = order_amount * (
                    self._market_info.maker_fee
                    if order.type == OrderType.LIMIT
                    else self._market_info.taker_fee
                )

                self._account.deposit(realized_pnl)
                self._account.deposit(-fee)

                del self._order_queue[order_id]

                transaction = {
                    "timestamp": timestamp,
                    "realized": {"pnl": realized_pnl, "fee": fee, },
                    "order": {
                        "price": order.price,
                        "quantity": order.quantity,
                        "side": order.side,
                        "type": order.type,
                    },
                }

                transactions.append(transaction)

        return transactions


class Engine:
    """
    Engine Object
    """
    def __init__(
            self,
            data: pd.DataFrame,
            initial_margin: float,
            market_info: MarketInfo,
            strategy_cls: type,
            strategy_config: Dict[str, Any],
    ):
        self._broker: Broker = Broker(market_info, initial_margin)
        self._strategy: Strategy = strategy_cls(strategy_config)

        indicator_df: pd.DataFrame = self._strategy.calculate(data).dropna()
        self._data_dict_list: List[
            Dict[str, float]
        ] = dataframe_as_list_of_dict(indicator_df)

        self._logger = Logger()

    @property
    def logger(self):
        return self._logger

    def run(self, use_tqdm=True, use_margin_call=False):
        """
        runs backtest

        :param use_tqdm: use tqdm progressbar or not
        :return: Logger object
        """
        iterator = range(len(self._data_dict_list) - 1)

        if use_tqdm:
            iterator = tqdm.tqdm(iterator)

        for i in iterator:
            data = self._data_dict_list[i]
            next_data = self._data_dict_list[i + 1]

            self._broker._account.update_portfolio_value(data["close"])

            account_info = self._broker.account_info

            unfilled_orders = {
                k: v.__dict__() for k, v in self._broker._order_queue.items()
            }

            state_dict = {
                "data": data,
                "account_info": account_info,
                "unfilled_orders": unfilled_orders,
            }

            actions = self._strategy.execute(state_dict)

            self._broker.apply_actions(
                actions, data["close"]
            )

            transactions = self._broker.fill_pending_orders(
                int(next_data["timestamp"]),
                next_data["open"],
                next_data["high"],
                next_data["low"],
            )

            account_info["timestamp"] = next_data["timestamp"]

            self._logger.add_transaction(transactions)
            self._logger.add_account_info(account_info)
            self._logger.add_unfilled_orders(self._broker._order_queue)

            if use_margin_call & (account_info["portfolio_value"] < 0):
                print("MARGIN CALL OCCURED, EXITING")
                break

        return self._logger