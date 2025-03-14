from datetime import date, datetime
from demeter import (
    TokenInfo,
    Actuator,
    Strategy,
    ChainType,
    MarketInfo,
    PeriodTrigger,
    simple_moving_average,
)
from demeter.result import performance_metrics, round_results
from demeter.uniswap import UniV3Pool, UniLpMarket
from demeter import Snapshot
from demeter import AtTimeTrigger
from datetime import date, datetime

usdc = TokenInfo(name="usdc", decimal=6)
eth = TokenInfo(name="eth", decimal=18)


def load_market_data(
    market: UniLpMarket,
    pool_address: str,
    start_date: date,
    end_date: date,
    data_path: str = "../data",
):
    market.data_path = data_path
    market.load_data(
        chain=ChainType.ethereum.name,
        contract_addr=pool_address,
        start_date=start_date,
        end_date=end_date,
    )
    return market


def run_baseline_strategy(market: UniLpMarket, initial_investment_usd):

    class NoProvisionStrategy(Strategy):
        def initialize(self):
            pass

        def on_bar(self, row_data: Snapshot):
            pass

    acturator = Actuator()
    acturator.broker.add_market(market)
    acturator.broker.set_balance(market.pool_info.token0, initial_investment_usd / 2)
    acturator.broker.set_balance(
        market.pool_info.token1,
        (initial_investment_usd / 2) / float(market.data.iloc[0].price),
    )
    acturator.strategy = NoProvisionStrategy()
    acturator.set_price(market.get_price_from_data())
    acturator.run(print_result=False)
    return acturator


def run_test_strategy(
    market: UniLpMarket, initial_investment_usd: float, strategy: Strategy
):

    actuator = Actuator()
    actuator.broker.add_market(market)
    actuator.broker.set_balance(market.pool_info.token0, initial_investment_usd / 2)
    actuator.broker.set_balance(
        market.pool_info.token1,
        (initial_investment_usd / 2) / float(market.data.iloc[0].price),
    )
    actuator.strategy = strategy
    actuator.set_price(market.get_price_from_data())

    actuator.run(print_result=False)
    actuator.strategy.finalize()
    return actuator
