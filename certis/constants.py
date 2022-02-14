class OrderSide:
    """
    OrderSide Object

    LONG: betting for upside
    SHORT: betting for downside
    """
    SHORT = -1
    LONG = 1
    SIDES = [SHORT, LONG]


class OrderType:
    """
    OrderType Object. Supported types are

    MARKET,
    LIMIT,
    STOP_MARKET,
    STOP_LOSS_MARKET,
    TAKE_PROFIT_MARKET
    """
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"

    STOP_LOSS_MARKET = "STOP_LOSS_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"

    STOP_ORDERS = [STOP_MARKET]
    LIMIT_ORDERS = [LIMIT]

    ORDERS = [MARKET, LIMIT, STOP_MARKET, STOP_LOSS_MARKET, TAKE_PROFIT_MARKET]
    MARKET_ORDERS = [MARKET, STOP_MARKET]
