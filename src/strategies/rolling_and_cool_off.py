import math
from demeter import Strategy, Snapshot, simple_moving_average
from datetime import timedelta
from decimal import Decimal

from .entities import StrategyParams


class RollingAndCoolOffStrategyParams(StrategyParams):
    percent_range: float
    rebalance_interval: timedelta

    def get_short_name(self):
        hours = self.rebalance_interval.total_seconds() / 3600

        return f"{self.name}_{str(self.percent_range).replace('.','dot')}_{hours:.0f}h_{int(self.initial_investment_usd)}USD"


class RollingAndCoolOffStrategy(Strategy):
    def __init__(self, market_key, strategy_params: RollingAndCoolOffStrategyParams):
        super().__init__()
        self.percent_range = Decimal(str(strategy_params.percent_range))
        self.rebalance_interval = strategy_params.rebalance_interval
        self.wait_until = None
        self.last_rebalance_time = None
        self.market_key = market_key

    def initialize(self):
        self.add_column(
            self.market_key,
            "rolling_avg",
            simple_moving_average(
                self.data[self.market_key].price, window=timedelta(hours=8)
            ),
        )

    def on_bar(self, snapshot: Snapshot):
        market = self.markets[self.market_key]
        current_time = snapshot.timestamp
        current_price = Decimal(str(snapshot.market_status[self.market_key].price))

        # Cooldown: Do not open positions during this period
        if self.wait_until and current_time < self.wait_until:
            if market.positions:
                market.remove_all_liquidity()
            return

        rolling_avg = self.data[self.market_key].loc[current_time]["rolling_avg"]
        if math.isnan(rolling_avg):
            rolling_avg = current_price
        rolling_avg = Decimal(str(rolling_avg))

        lower_bound = rolling_avg * (1 - self.percent_range)
        upper_bound = rolling_avg * (1 + self.percent_range)

        has_position = len(market.positions) > 0

        # If we have a position, check if price crosses bounds
        if has_position:
            current_position = next(iter(market.positions.values()))
            lower_price = current_position.lower_price
            upper_price = current_position.upper_price

            if current_price > upper_price:
                market.remove_all_liquidity()
                market.even_rebalance(current_price)
                market.add_liquidity(lower_bound, upper_bound)
                self.last_rebalance_time = current_time
                self.wait_until = None
                return

            elif current_price < lower_price:
                market.remove_all_liquidity()
                market.even_rebalance(current_price)
                self.wait_until = current_time + timedelta(hours=12)
                self.last_rebalance_time = current_time
                return

        needs_rebalance = (self.last_rebalance_time is None) or (
            (current_time - self.last_rebalance_time) >= self.rebalance_interval
        )
        # Open new position only if there is no current position, not in cooldown, and rebalance interval elapsed
        if needs_rebalance:
            market.remove_all_liquidity()
            market.even_rebalance(current_price)
            market.add_liquidity(lower_bound, upper_bound)
            self.last_rebalance_time = current_time

    def finalize(self):
        market = self.markets[self.market_key]
        if market.positions:
            market.remove_all_liquidity()
