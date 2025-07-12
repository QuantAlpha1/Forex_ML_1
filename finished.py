
import sys
import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime, time as dt_time, timedelta
import pytz
import numpy as np
from typing import Tuple, Optional

class RiskBreach(Exception):
    """Custom exception for risk management violations"""
    def __init__(self, message="Risk threshold breached"):
        self.message = message
        super().__init__(self.message)

class RiskManager:
    """Handles all risk-related checks and operations"""
    def __init__(self, config: dict, scalper=None):
        self.config = config
        self.scalper = scalper # Reference to Scalper instance
        self.drawdown_start_balance = None
        
    
    def check_drawdown(self, current_equity: float) -> None:
        """Safe drawdown check with PNL awareness"""
        if self.drawdown_start_balance is None:
            self.drawdown_start_balance = current_equity
            return
            
        # Safe PNL check if scalper reference exists
        if hasattr(self, 'scalper') and self.scalper is not None:
            try:
                daily_pnl = self.scalper.calculate_daily_pnl()
                if daily_pnl < -0.5 * (self.config['max_daily_drawdown']/100) * self.drawdown_start_balance:
                    print(f"PNL Warning: ${daily_pnl:.2f}")
            except AttributeError:
                print("Warning: calculate_daily_pnl not available")
            except Exception as e:
                print(f"PNL check failed: {str(e)}")

        drawdown_pct = (self.drawdown_start_balance - current_equity) / self.drawdown_start_balance * 100
        if drawdown_pct >= self.config['max_daily_drawdown']:
            raise RiskBreach(f"Drawdown limit: {drawdown_pct:.2f}%")
            
    def reset_daily_drawdown(self):
        """Reset at start of new trading day"""
        self.drawdown_start_balance = None
        print("Daily drawdown reset")

    def check_position_count(self, current_count: int) -> None:
        if current_count >= self.config.get('max_simultaneous_trades', 5):
            raise RiskBreach(f"Max positions reached: {current_count}")

class Scalper:
    """Production-Ready MT5 Scalper for EURUSD and USDJPY"""
    
    def __init__(self, risk_percent: float = 0.5, risk_reward: float = 1.5):
        self._verify_broker_conditions() 
        self.risk_manager = RiskManager(self.config, scalper=self)
        self.last_trade_time = {}
        self.config = {
            'risk_percent': risk_percent,
            'risk_reward': risk_reward,
            'max_daily_drawdown': 2.0,  # Percentage
            'drawdown_start_time': None,
            'magic_number': 234000,
            'deviation': 10,
            'max_spread': 2.0,
            'trading_hours': {
                'start': dt_time(8, 0, tzinfo=pytz.timezone('Europe/London')),
                'end': dt_time(17, 0, tzinfo=pytz.timezone('Europe/London'))
            },
            'max_trade_duration': 15,  # minutes
            'trail_start': 10,  # pips
            'trail_step': 5     # pips
        }

        self.risk_manager = RiskManager(self.config, scalper=self)  # Pass self reference

        self.min_acceptable_volatility = {
            'EURUSD': 0.0003,
            'USDJPY': 0.03  
        }
        self.max_acceptable_volatility = {
            'EURUSD': 0.002,
            'USDJPY': 0.20
        }


        self.trade_log = [] 
        
        if not mt5.initialize():
            raise ConnectionError("Failed to initialize MT5")
        print("MT5 initialized successfully")
        
        # Preload symbol info
        self.symbols = self._setup_symbols(["EURUSD", "USDJPY"])

    def __del__(self):
        mt5.shutdown()
        print("MT5 connection closed")

    def _setup_symbols(self, symbols: list) -> dict:
        """Preload symbol properties with enhanced validation and error handling"""
        symbol_info = {}
        
        for symbol in symbols:
            # Symbol selection with retry logic
            if not mt5.symbol_select(symbol, True):
                print(f"Failed to select {symbol} - skipping")
                continue
                
            # Get symbol info with error handling
            try:
                info = mt5.symbol_info(symbol)
                if info is None:
                    print(f"Could not get symbol info for {symbol} - skipping")
                    continue
                    
                # Basic symbol properties
                symbol_info[symbol] = {
                    'point': info.point,
                    'digits': info.digits,
                    'volume_step': info.volume_step,
                    'volume_min': info.volume_min,
                    'volume_max': info.volume_max,
                    'trade_allowed': info.trade_allowed,
                    'pip_value': 10  # Default for non-JPY pairs
                }
                
                # JPY-specific handling with tick validation
                if "JPY" in symbol:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick and tick.bid > 0:
                        symbol_info[symbol]['pip_value'] = 1000 / tick.bid
                    else:
                        print(f"Invalid tick data for {symbol}, using default pip value")
                        
                # Additional validation for critical values
                if symbol_info[symbol]['point'] <= 0:
                    print(f"Invalid point value for {symbol} - skipping")
                    del symbol_info[symbol]
                    continue
                    
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                continue
                
        return symbol_info

    def _is_optimal_trading_time(self):
        """Filter for high probability trading windows"""
        now = datetime.now(pytz.timezone('Europe/London'))
        current_time = now.time()
        current_hour = current_time.hour
        
        # London open (8-9am) and US open (2-3pm London time)
        optimal_windows = [
            (dt_time(8, 0), dt_time(9, 0)),   # London open
            (dt_time(14, 0), dt_time(15, 0)),  # NY open
            (dt_time(15, 30), dt_time(16, 30)) # NY mid-session
        ]
        
        return any(start <= current_time < end for start, end in optimal_windows)

    def _is_trading_hours(self) -> bool:
        """Robust trading hour check with timezone awareness"""
        london_tz = pytz.timezone('Europe/London')
        now = datetime.now(london_tz).time()
        return self.config['trading_hours']['start'] <= now <= self.config['trading_hours']['end']

    def _is_london_open(self) -> bool:
        """Check if current time is within London open window (8-9am London)"""
        london_tz = pytz.timezone('Europe/London')
        now = datetime.now(london_tz).time()
        return dt_time(7, 55) <= now <= dt_time(9, 5)  # 5-min buffer

    def _is_ny_close(self) -> bool:
        """Check if current time is within NY close window (4-5pm London/11am-12pm NY)"""
        london_tz = pytz.timezone('Europe/London') 
        now = datetime.now(london_tz).time()
        return dt_time(16, 0) <= now <= dt_time(17, 5)  # 5-min buffer

    def calculate_position_size(self, symbol: str, entry: float, stop_loss: float) -> float:
        """Risk-based position sizing with contract size"""
        account = mt5.account_info()
        if not account:
            return 0.0
        
        risk_amount = account.balance * (self.config['risk_percent'] / 100)
        point = self.symbols[symbol]['point']
        risk_points = abs(entry - stop_loss) / point
        
        # Currency-specific pip value
        if "JPY" in symbol:
            pip_value = 1000 / entry  # JPY pairs
        else:
            pip_value = 10  # Non-JPY pairs
        
        size = risk_amount / (risk_points * point * pip_value)
        return self._normalize_volume(symbol, size)

    def calculate_atr(self, symbol: str, timeframe: int = mt5.TIMEFRAME_M5, period: int = 14) -> float:
        """Efficient ATR calculation with error handling"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period+1)
            if rates is None or len(rates) < period+1:
                return 0.0
            
            df = pd.DataFrame(rates)
            df['prev_close'] = df['close'].shift(1)
            
            # Calculate True Range components
            hl = (df['high'] - df['low']).abs()
            hc = (df['high'] - df['prev_close']).abs()
            cl = (df['prev_close'] - df['low']).abs()
            
            # Get max of the three components
            df['tr'] = pd.concat([hl, hc, cl], axis=1).max(axis='columns')
            
            return df['tr'].tail(period).mean()
        except Exception as e:
            print(f"ATR calculation error for {symbol}: {str(e)}")
            return 0.0

    def _validate_atr(self, atr: float, symbol: str) -> bool:
        """More robust volatility check"""
        if atr <= 0:
            return False
        pair = symbol[:6]
        min_atr = self.min_acceptable_volatility.get(pair, 0.0003)
        max_atr = self.max_acceptable_volatility.get(pair, 0.002)
        return min_atr <= atr <= max_atr

    def entry_signal(self, symbol: str) -> Optional[str]:
        """Enhanced multi-factor entry signal with weighted confirmations"""
        try:
            # Get data for multiple timeframes
            m1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 15)
            m5_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 10)
            
            if m1_rates is None or m5_rates is None or len(m1_rates) < 15 or len(m5_rates) < 10:
                return None
                
            # M1 Breakout logic
            m1_closes = [r['close'] for r in m1_rates]
            m1_breakout_buy = m1_closes[-1] > max(m1_closes[-6:-1])
            m1_breakout_sell = m1_closes[-1] < min(m1_closes[-6:-1])
            
            # M5 Trend filter
            m5_closes = [r['close'] for r in m5_rates]
            m5_ma = sum(m5_closes[-5:])/5  # Simple 5-period MA
            m5_trend_filter_buy = m5_closes[-1] > m5_ma
            m5_trend_filter_sell = m5_closes[-1] < m5_ma
            
            # Volume spike confirmation
            volume_confirmed = False
            if len(m1_rates) > 10 and 'real_volume' in m1_rates[0]._asdict():
                avg_volume = sum(r['real_volume'] for r in m1_rates[-10:-1])/9
                volume_spike = m1_rates[-1]['real_volume'] > avg_volume * 1.5
                volume_confirmed = volume_spike
            
            # Price action confirmation with confidence
            price_action_signal, pa_confidence = self._detect_price_action(m1_rates)
            
            # Order flow confirmation
            buy_flow, sell_flow = self._order_flow_confirmation(symbol)

            # Only consider order flow if we have meaningful data
            order_flow_weight = 0.0
            if buy_flow > 0.6 or sell_flow > 0.6:  # Strong signal
                order_flow_weight = 0.25
            elif buy_flow > 0.4 or sell_flow > 0.4:  # Moderate signal
                order_flow_weight = 0.15
            # Else: weak signal gets no weight
                
            # Weighted decision making
            buy_score = 0
            sell_score = 0
            
            atr = self.calculate_atr(symbol)
            if not self._validate_atr(atr, symbol):  # PROPER VALIDATION
                print(f"ATR validation failed for {symbol}: {atr:.5f}")
                return None
        
            daily_atr = atr * (24*12)
            if not self.is_acceptable_volatility(symbol, daily_atr):
                return None
                
            # Weighted decision making
            buy_score = 0
            sell_score = 0
            
            # Breakout (40% weight)
            if m1_breakout_buy:
                buy_score += 0.4
            if m1_breakout_sell:
                sell_score += 0.4

            # Order flow contribution
            if order_flow_weight > 0:
                if buy_flow > sell_flow:
                    buy_score += order_flow_weight * (buy_flow - sell_flow)  # Only the differential
                else:
                    sell_score += order_flow_weight * (sell_flow - buy_flow)
                
            # Trend filter (20% weight)
            if m5_trend_filter_buy:
                buy_score += 0.2
            if m5_trend_filter_sell:
                sell_score += 0.2
                
            # Volume (15% weight)
            if volume_confirmed:
                if m1_breakout_buy:
                    buy_score += 0.15
                if m1_breakout_sell:
                    sell_score += 0.15
                    
            # Price action (15% weight)
            if price_action_signal == 'buy':
                buy_score += 0.15 * pa_confidence
            elif price_action_signal == 'sell':
                sell_score += 0.15 * pa_confidence
                
            # Order flow (10% weight)
            if buy_flow:
                buy_score += 0.1
            if sell_flow:
                sell_score += 0.1
                
            # Only enter if we have strong confirmation (>= 0.75 score)
            if buy_score >= 0.75 and buy_score > sell_score:
                return 'buy'
            elif sell_score >= 0.75 and sell_score > buy_score:
                return 'sell'
                
        except Exception as e:
            print(f"Signal error for {symbol}: {str(e)}")
        return None
    
    def _detect_price_action(self, rates: list) -> Tuple[Optional[str], float]:
        """Detect price action patterns with confidence scoring
        Returns:
            tuple: (signal_direction, confidence_score) where score is 0-1
        """
        if len(rates) < 3:
            return None, 0.0
            
        current = rates[-1]
        prev = rates[-2]
        prev_prev = rates[-3]
        
        confidence = 0.0
        signal = None
        
        # Calculate basic candle properties
        body_size = abs(current['open'] - current['close'])
        total_range = current['high'] - current['low']
        
        if total_range > 0:  # Avoid division by zero
            # Pinbar detection with confidence scoring
            upper_wick = current['high'] - max(current['open'], current['close'])
            lower_wick = min(current['open'], current['close']) - current['low']
            
            # Bullish pinbar
            if (lower_wick >= 2 * body_size and 
                lower_wick >= 0.6 * total_range):
                signal = 'buy'
                confidence = min(0.9, lower_wick / (total_range * 1.5))
                
            # Bearish pinbar
            elif (upper_wick >= 2 * body_size and 
                upper_wick >= 0.6 * total_range):
                signal = 'sell'
                confidence = min(0.9, upper_wick / (total_range * 1.5))
        
        # Engulfing pattern detection with confidence
        # Bullish engulfing
        if (current['close'] > current['open'] and
            prev['close'] < prev['open'] and
            current['open'] < prev['close'] and
            current['close'] > prev['open']):
            
            engulfing_size = current['close'] - current['open']
            prev_size = prev['open'] - prev['close']
            engulfing_ratio = engulfing_size / prev_size if prev_size > 0 else 1
            
            if engulfing_ratio > 1.2:  # Only consider strong engulfing
                signal = 'buy'
                confidence = max(confidence, min(0.8, engulfing_ratio / 2))
                
        # Bearish engulfing
        elif (current['close'] < current['open'] and
            prev['close'] > prev['open'] and
            current['open'] > prev['close'] and
            current['close'] < prev['open']):
            
            engulfing_size = current['open'] - current['close']
            prev_size = prev['close'] - prev['open']
            engulfing_ratio = engulfing_size / prev_size if prev_size > 0 else 1
            
            if engulfing_ratio > 1.2:
                signal = 'sell'
                confidence = max(confidence, min(0.8, engulfing_ratio / 2))
        
        return signal, confidence

    def place_trade(self, symbol: str, signal: str, atr: Optional[float] = None) -> bool:
        """Production-grade trade execution"""
        

        # Initial checks - no tick required
        if not self._is_trading_hours():
            return False
        if not self.symbols[symbol]['trade_allowed']:
            print(f"Trading not allowed for {symbol}")
            return False
        if signal not in ['buy', 'sell']:
            print(f"Invalid signal: {signal}")
            return False
        if self.check_open_positions(symbol):
            return False

        # Get account info first
        account = mt5.account_info()
        if not account:
            print("Failed to get account info")
            return False
        
        

        # First tick retrieval
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"Failed to get tick for {symbol}")
            return False
            
        
        # Spread check with tolerance
        current_spread = (tick.ask - tick.bid) / self.symbols[symbol]['point']
        if current_spread > self.config['max_spread'] * 1.5:
            print(f"Spread too wide: {current_spread:.1f} points")
            return False
        
        # ATR checks
        atr = self.calculate_atr(symbol) if atr is None else atr
        if not self._validate_atr(atr, symbol):
            return False

        # Slippage check
        expected_price = tick.ask if signal == 'buy' else tick.bid
        if abs(tick.last - expected_price) > 3*self.symbols[symbol]['point']:
            print(f"High slippage detected ({tick.last} vs {expected_price}) - aborting")
            return False

        # Entry price using same tick
        entry = expected_price
        stop_loss, take_profit = self.calculate_sl_tp(symbol, entry, signal == 'buy', atr)
        
        if not self.validate_trade(entry, stop_loss, take_profit):
            return False   

        # Session-based risk adjustment 
        original_risk = self.config['risk_percent']
        
        if self._is_london_open():
            # Increase aggression during London open (8-9am London)
            self.config['risk_percent'] = min(original_risk * 1.3, 1.0)  # Cap at 1%
            print(f"London open active - risk adjusted to {self.config['risk_percent']:.2f}%")
        elif self._is_ny_close():
            # Reduce risk during NY close (4-5pm London)
            self.config['risk_percent'] = min(max(self.config['risk_percent'], 0.1), 2.0)  # Keep between 0.1-2%
            print(f"NY close approaching - risk reduced to {self.config['risk_percent']:.2f}%")

        # Dynamic risk adjustment
        original_risk = self.config['risk_percent']
        adjusted_risk = self._adjust_risk_based_on_volatility(symbol)
        
        if adjusted_risk != original_risk:
            print(f"Volatility adjustment: Risk changed from {original_risk}% to {adjusted_risk}%")
            self.config['risk_percent'] = adjusted_risk  # Temporary adjustment
        
        # SINGLE position size calculation
        size = self.calculate_position_size(symbol, entry, stop_loss)
        
        # Restore original risk immediately
        self.config['risk_percent'] = min(max(self.config['risk_percent'], 0.1), 2.0)  # Keep between 0.1-2%
            
        if size <= 0:
            print(f"Invalid position size for {symbol}: {size:.2f} lots")
            return False
        
        # Validate position size
        if not self.validate_position_size(symbol, size):
            print(f"Invalid size {size:.2f} for {symbol}. Min: {self.symbols[symbol]['volume_min']}, Max: {self.symbols[symbol]['volume_max']}")
            return False

        # Risk management check
        current_positions = len([p for p in mt5.positions_get() if p.magic == self.config['magic_number']])
        self.risk_manager.check_position_count(current_positions)

        # Define order type
        order_type = mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL
        
        # Margin check
        margin_req = mt5.order_calc_margin(order_type, symbol, size, entry)
        if margin_req > account.margin_free:
            print(f"Insufficient margin: {margin_req:.2f} required, {account.margin_free:.2f} available")
            return False

        # Execute trade
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": size,
            "type": order_type,
            "price": entry,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": self.config['deviation'],
            "magic": self.config['magic_number'],
            "comment": "Scalper",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Add retry logic
        max_retries = 3
        for attempt in range(max_retries):
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                break
            time.sleep(1)
        else:
            print(f"Order failed after {max_retries} attempts")
            return False

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed with retcode {result.retcode}")
            return False
            
        # Enhanced partial fill handling
        if result.volume_executed < result.volume_requested:
            print(f"Partial fill: {result.volume_executed}/{size} lots")
            self._handle_partial_fill(result, symbol, signal)
            
        # Log successful trade
        self.trade_log.append({
            'time': datetime.now(),
            'symbol': symbol,
            'direction': signal,
            'size': size,
            'entry': entry,
            'sl': stop_loss,
            'tp': take_profit,
            'status': 'open',
            'ticket': result.order,
            'fill_ratio': result.volume_executed/size if size > 0 else 1.0
        })

        # Log trade event
        self.log_trade_event('open', {
            'symbol': symbol,
            'size': size,
            'price': entry
        })
        
        print(f"Trade executed: {symbol} {signal} at {entry} ({size:.2f} lots)")
        return True

    def calculate_sl_tp(self, symbol: str, entry: float, is_long: bool, atr: float) -> Tuple[float, float]:
        """ATR-based stops with minimum distance"""
        min_dist = 10 * self.symbols[symbol]['point']
        sl_dist = max(1.5 * atr, min_dist)
        tp_dist = sl_dist * self.config['risk_reward']
        
        if is_long:
            return (entry - sl_dist, entry + tp_dist)
        return (entry + sl_dist, entry - tp_dist)

    def _normalize_volume(self, symbol: str, volume: float) -> float:
        """Safe volume normalization"""
        step = self.symbols[symbol]['volume_step']
        return round(volume / step) * step

    def validate_trade(self, entry: float, sl: float, tp: float) -> bool:
        """Risk-reward validation"""
        try:
            risk = abs(entry - sl)
            reward = abs(entry - tp)
            return reward / risk >= self.config['risk_reward']
        except ZeroDivisionError:
            return False

    def validate_position_size(self, symbol, size):
        """Validate position size against symbol's min/max limits"""
        if symbol not in self.symbols:
            return False
            
        info = self.symbols[symbol]
        
        # Check if volume limits exist in the symbol info
        if 'volume_min' not in info or 'volume_max' not in info:
            # Fallback to MT5's symbol info if not in our cache
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                return (size >= symbol_info.volume_min and 
                        size <= symbol_info.volume_max)
            return False
            
        return size >= info['volume_min'] and size <= info['volume_max']

    def check_open_positions(self, symbol: str) -> bool:
        """Check positions with magic number"""
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return False
        return any(pos.magic == self.config['magic_number'] for pos in positions)

    def manage_trades(self):
        """Professional trade management with ATR flow"""
        current_positions = len([p for p in mt5.positions_get() if p.magic == self.config['magic_number']])
        self.risk_manager.check_position_count(current_positions)

        # First check for any closed positions to update logs
        self.check_closed_positions()
        
        # Pre-calculate ATR for all symbols once
        atr_cache = {
            symbol: {
                'raw': self.calculate_atr(symbol),
                'daily': self.calculate_atr(symbol) * (24*12)
            } for symbol in self.symbols
        }
        
        positions = mt5.positions_get()
        if positions is None:
            return

        london_tz = pytz.timezone('Europe/London')
        current_time = datetime.now(london_tz)

        for pos in positions:
            if pos.magic != self.config['magic_number']:
                continue

            try:
                symbol = pos.symbol
                pos_type = pos.type
                
                # Get current price with safety check
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    continue
                current_price = tick.bid if pos_type == mt5.ORDER_TYPE_BUY else tick.ask
                
                # Get pre-calculated ATR
                atr = atr_cache[symbol]
                daily_atr = atr['raw'] * (24*12)  # Convert to daily
                
                # Volatility check
                if not self.is_acceptable_volatility(symbol, daily_atr):
                    self.close_position(pos)
                    continue

                # Time-based exit
                open_time = datetime.fromtimestamp(pos.time, tz=london_tz)
                duration = (current_time - open_time).total_seconds() / 60
                if duration > self.config['max_trade_duration']:
                    self.close_position(pos)
                    continue

                # Risk/reward calculation
                risk = abs(pos.price_open - pos.sl)
                profit = abs(current_price - pos.price_open)
                rr_ratio = profit / risk if risk > 0 else 0

                # Partial profit taking
                if rr_ratio >= 1.5 and pos.volume > 0.02:
                    self.close_partial_position(pos, pos.volume * 0.5)

                # Trailing stop with ATR flow
                self.update_trailing_stop(
                    position=pos,
                    current_price=current_price,
                    atr=atr['raw']  # Correctly passing just the float value
                )

            except Exception as e:
                print(f"Trade management error for {pos.symbol}: {str(e)}")

    def update_trailing_stop(self, position, current_price, atr: Optional[float] = None):
        """Enhanced trailing stop implementation with ATR"""
        symbol = position.symbol
        point = self.symbols[symbol]['point']
        pip_value = point * 10
        
        if atr is None:
            atr = self.calculate_atr(symbol)  # Fallback
        
        profit_pips = abs(current_price - position.price_open) / pip_value
        
        # Dynamic ATR trailing when profit is good
        if profit_pips >= self.config['trail_start'] * 2:
            dynamic_factor = min(2.0, 1.0 + (profit_pips/10))
            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = current_price - (atr * dynamic_factor)
                # Ensure new SL is valid and better than current
                if new_sl > max(position.sl, position.price_open):
                    self.modify_sl(position, new_sl)
            else:  # SELL position
                new_sl = current_price + (atr * dynamic_factor)
                if new_sl < min(position.sl, position.price_open) or position.sl == 0:
                    self.modify_sl(position, new_sl)
        else:
            # Default fixed trailing
            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = current_price - self.config['trail_step'] * pip_value
                if new_sl > max(position.sl, position.price_open):
                    self.modify_sl(position, new_sl)
            else:  # SELL position
                new_sl = current_price + self.config['trail_step'] * pip_value
                if new_sl < min(position.sl, position.price_open) or position.sl == 0:
                    self.modify_sl(position, new_sl)

    def close_position(self, position):
        request = self._close_request(position, position.volume)
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self._update_trade_log(position.ticket, 'closed', result.price)
        print(f"Daily PNL after close: ${self.calculate_daily_pnl():.2f}")

    def log_trade_event(self, event_type: str, details: dict):
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'event': event_type,
            'symbol': details.get('symbol'),
            'price': details.get('price'),
            'size': details.get('size'),
            'equity': mt5.account_info().equity if mt5.initialize() else None,
            'daily_pnl': self.calculate_daily_pnl()
        }

        self.trade_log.append(log_data) 

    def calculate_daily_pnl(self) -> float:
        """Calculate today's realized PnL from closed trades in trade_log"""
        today = datetime.now().date()
        pnl = 0.0
        for trade in self.trade_log:
            if trade.get('status') == 'closed' and trade.get('time', datetime.now()).date() == today:
                pnl += self.calculate_trade_profit(trade)
        return pnl

    def close_partial_position(self, position, volume):
        request = self._close_request(position, volume)
        result = mt5.order_send(request)  # <-- This line was missing
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.log_trade_event('partial_close', {
                'symbol': position.symbol,
                'size': volume,
                'price': result.price
            })
        print(f"Daily PNL after partial close: ${self.calculate_daily_pnl():.2f}")
        return result.retcode == mt5.TRADE_RETCODE_DONE 
            
    def check_closed_positions(self):
        closed = mt5.history_deals_get(datetime.now() - timedelta(minutes=30), datetime.now())
        for deal in closed:
            if deal.entry == mt5.DEAL_ENTRY_OUT:
                self._update_trade_log(deal.position_id, 'closed', deal.price)

    def _close_request(self, position, volume):
        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": position.symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(position.symbol).ask if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).bid,
            "deviation": self.config['deviation'],
            "magic": self.config['magic_number'],
            "comment": "Exit",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

    def modify_sl(self, position, new_sl):
        """Modify stop loss"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "symbol": position.symbol,
            "sl": new_sl,
            "tp": position.tp,
            "deviation": self.config['deviation'],
            "magic": self.config['magic_number'],
            "comment": "Trailing stop",
            "type_time": mt5.ORDER_TIME_GTC,
        }
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE

    def _update_trade_log(self, ticket, status, exit_price=None):
     for trade in self.trade_log:
         if trade['ticket'] == ticket:
             trade['status'] = status
             if exit_price is not None:
                 trade['exit_price'] = exit_price
             break

    def performance_report(self):
        """Enhanced performance analytics"""
        closed_trades = [t for t in self.trade_log if t['status'] == 'closed']
        if not closed_trades:
            print("No closed trades")
            return

        # Add daily PNL display
        daily_pnl = self.calculate_daily_pnl()
        print(f"\nDaily P&L: ${daily_pnl:.2f}")
            
        wins = [t for t in closed_trades if self.calculate_trade_profit(t) > 0]
        losses = [t for t in closed_trades if self.calculate_trade_profit(t) <= 0]
        
        win_rate = len(wins) / len(closed_trades)
        avg_win = np.mean([self.calculate_trade_profit(t) for t in wins]) if wins else 0
        avg_loss = np.mean([abs(self.calculate_trade_profit(t)) for t in losses]) if losses else 0
        profit_factor = (win_rate * avg_win) / ((1 - win_rate) * avg_loss) if avg_loss > 0 else float('inf')
        
        print(f"\nPerformance Report ({len(closed_trades)} trades)")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Expectancy: ${(win_rate * avg_win - (1 - win_rate) * avg_loss):.2f}")

    def _order_flow_confirmation(self, symbol: str) -> Tuple[float, float]:
        """Comprehensive order flow analysis with validation
        Returns:
            tuple: (buy_strength, sell_strength) scores between 0-1
            Returns (0.0, 0.0) if analysis fails or data is unavailable
        """
        # Initialize default neutral values
        buy_strength = 0.0
        sell_strength = 0.0

        try:
            # Get current market depth with retry logic
            max_retries = 3
            retry_delay = 0.5
            depth = None
            
            for attempt in range(max_retries):
                try:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        time.sleep(retry_delay)
                        continue
                        
                    depth = mt5.market_book_get(symbol) if hasattr(mt5, 'market_book_get') else None
                    if depth is not None:
                        break
                except Exception as e:
                    print(f"Order flow retry attempt {attempt + 1} failed: {str(e)}")
                time.sleep(retry_delay)
            
            if depth is None or len(getattr(depth, 'ask', [])) < 5 or len(getattr(depth, 'bid', [])) < 5:
                print(f"Market depth unavailable for {symbol}")
                return 0.0, 0.0  # Neutral if no depth data
                
            # Get multiple levels for more robust analysis
            ask_levels = depth.ask[:5]  # Top 5 ask levels
            bid_levels = depth.bid[:5]  # Top 5 bid levels
            
            # Calculate total volumes with safety checks
            try:
                total_ask_vol = sum(level[1] for level in ask_levels if len(level) > 1)
                total_bid_vol = sum(level[1] for level in bid_levels if len(level) > 1)
            except (IndexError, TypeError):
                total_ask_vol = 0.0
                total_bid_vol = 0.0
            
            # Calculate volume deltas with safety checks
            bid_deltas = []
            ask_deltas = []
            try:
                for i in range(1, min(5, len(bid_levels))):
                    if len(bid_levels[i]) > 1 and bid_levels[i][1] > 0:
                        bid_deltas.append((bid_levels[i-1][1] - bid_levels[i][1]) / bid_levels[i][1])
                for i in range(1, min(5, len(ask_levels))):
                    if len(ask_levels[i]) > 1 and ask_levels[i][1] > 0:
                        ask_deltas.append((ask_levels[i-1][1] - ask_levels[i][1]) / ask_levels[i][1])
            except (IndexError, TypeError):
                pass
            
            avg_bid_delta = sum(bid_deltas)/len(bid_deltas) if bid_deltas else 0
            avg_ask_delta = sum(ask_deltas)/len(ask_deltas) if ask_deltas else 0
            
            # Calculate price momentum with safety checks
            try:
                bid_price_momentum = (bid_levels[0][0] - bid_levels[-1][0]) / bid_levels[-1][0] if bid_levels[-1][0] > 0 else 0
                ask_price_momentum = (ask_levels[-1][0] - ask_levels[0][0]) / ask_levels[0][0] if ask_levels[0][0] > 0 else 0
            except (IndexError, TypeError):
                bid_price_momentum = 0
                ask_price_momentum = 0
            
            # Combined scoring (0-1 range) with epsilon for division safety
            try:
                total_volume = (total_ask_vol + total_bid_vol) or 1e-8  # Prevent division by zero
                buy_strength = min(1.0, max(0.0, 
                    (0.4 * (total_bid_vol / total_volume) +
                    (0.3 * avg_bid_delta) +
                    (0.3 * bid_price_momentum * 100)
                ))
                )
                
                sell_strength = min(1.0, max(0.0,
                    (0.4 * (total_ask_vol / total_volume) +
                    (0.3 * avg_ask_delta) +
                    (0.3 * ask_price_momentum * 100)
                ))
                )

            except Exception:
                buy_strength, sell_strength = 0.0, 0.0

            # Add liquidity validation
            if total_bid_vol < 100 or total_ask_vol < 100:  # Minimum volume threshold
                return 0.0, 0.0
            
            # Validate against current price movement
            try:
                if len(bid_levels) > 0 and len(ask_levels) > 0 and len(bid_levels[0]) > 0 and len(ask_levels[0]) > 0:
                    spread = ask_levels[0][0] - bid_levels[0][0]
                    if spread > 10 * self.symbols[symbol]['point']:  # Wide spread
                        buy_strength *= 0.7
                        sell_strength *= 0.7
            except Exception:
                pass

        except Exception as e:
            print(f"Order flow analysis failed for {symbol}: {str(e)}")
            # buy_strength, sell_strength = 0.0, 0.0
        
        # return buy_strength, sell_strength
        return 0.0, 0.0  # Neutral if any error occurs

    def calculate_trade_profit(self, trade):
        if trade['status'] != 'closed':
            return 0
            
        # Calculate price difference
        if trade['direction'] == 'buy':
            price_diff = trade['exit_price'] - trade['entry']
        else:  # sell
            price_diff = trade['entry'] - trade['exit_price']
        
        # JPY-specific calculation (convert yen profit to account currency)
        if "JPY" in trade['symbol']:
            # Profit in JPY: (price_diff / 0.01) * 1000 * size
            # Then convert to account currency (USD) using exit price
            jpy_profit = price_diff * 100000  # Profit in JPY
            return jpy_profit / trade['exit_price'] * trade['size']
        
        # Standard pairs (EURUSD)
        return price_diff * 100000 * trade['size']  # 100,000 units per standard lot

    def _adjust_risk_based_on_volatility(self, symbol: str) -> float:
        """Dynamically adjusts risk based on current volatility"""
        atr = self.calculate_atr(symbol)
        base_risk = self.config['risk_percent']  # Your existing risk setting
        
        # Define thresholds directly in the method
        HIGH_VOLATILITY_THRESHOLD = 0.003
        RISK_REDUCTION_FACTOR = 0.7  # Reduce to 70% of original risk
        
        if atr > HIGH_VOLATILITY_THRESHOLD:
            return base_risk * RISK_REDUCTION_FACTOR
        return base_risk  # Default unchanged risk

    def run(self):
        """Add time synchronization"""
        """Robust main trading loop"""
        print("Starting scalping bot...")
        print(f"Pandas: {pd.__version__}")
        print(f"MT5: {mt5.__version__}")
        last_reconnect = time.time()
        last_day_check = datetime.now()
        trade_count = 0
        
        while True:
            try:
                current_time = datetime.now()
                
                # Daily reset check
                if current_time.date() != last_day_check.date():
                    self.risk_manager.reset_daily_drawdown()
                    last_day_check = current_time
                    
                # Connection management
                if not mt5.initialize() or not mt5.terminal_info().connected:
                    self.reconnect()
                    
                if time.time() - last_reconnect > 3600:  # Hourly reconnect
                    mt5.shutdown()
                    mt5.initialize()
                    last_reconnect = time.time()

                # Sync with broker time
                broker_time = mt5.symbol_info_tick(symbol).time
                local_drift = datetime.now() - broker_time
                if abs(local_drift) > timedelta(seconds=2):
                    print(f"Time drift detected: {local_drift}")
                    
                # Handle DST transitions
                london_tz = pytz.timezone('Europe/London')
                if london_tz.localize(datetime.now()).dst() != self.last_dst_state:
                    print("Daylight savings change detected")
                    self.last_dst_state = not self.last_dst_state

                # Check if we should trade (optimal time or not strict)
                should_trade = (self._is_optimal_trading_time() or 
                            not self.config.get('strict_optimal_time', False))
                
                if should_trade:
                    print(f"{current_time} Optimal trading time detected")
                    
                    # Trading signals and execution
                    account = mt5.account_info()
                    if account:
                        self.risk_manager.check_drawdown(account.equity)
                        
                        for symbol in self.symbols:
                            signal = self.entry_signal(symbol)
                            if signal:
                                print(f"{current_time} {symbol} {signal.upper()} signal")
                                self.place_trade(symbol, signal)
                    else:
                        print("Failed to retrieve account info")
                
                # Trade management
                self.manage_trades()
                
                # Periodic reporting
                trade_count += 1
                if trade_count % 10 == 0:
                    current_pnl = self.calculate_daily_pnl()
                    if abs(current_pnl) > 1000:  # Example threshold
                        print(f"Significant daily PNL movement: ${current_pnl:.2f}")
                    self.performance_report()
                    
                time.sleep(5)
                
            except RiskBreach as e:
                print(f"RISK MANAGEMENT TRIGGERED: {str(e)}")
                self.emergency_stop()
                break
                
            except KeyboardInterrupt:
                print("Stopped by user")
                break
                
            except Exception as e:
                print(f"System error: {str(e)}")
                self.reconnect()
                time.sleep(30)

    def _verify_broker_conditions(self):
        """Verify broker conditions before trading"""
        if not mt5.initialize():
            raise ConnectionError("Failed to initialize MT5")
        
        account_info = mt5.account_info()
        if account_info is None:
            raise ConnectionError("Failed to retrieve account info")
        
        if not account_info.trade_allowed:
            raise PermissionError("Trading is not allowed on this account")
        
        print("Broker conditions verified successfully")

    def emergency_stop(self):
        """Close all positions and cancel orders"""
        print("Executing emergency stop...")
        for pos in mt5.positions_get():
            if pos.magic == self.config['magic_number']:
                self.close_position(pos)
        
        for order in mt5.orders_get():
            if order.magic == self.config['magic_number']:
                mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket})
        
        self.risk_manager.reset_daily_drawdown()
        print("Emergency stop complete")

    
    def reconnect(self):
        """Reconnect to MetaTrader5 terminal"""
        try:
            mt5.shutdown()
        except Exception:
            pass
        finally:
            time.sleep(2)
            if not mt5.initialize():
                raise ConnectionError("Failed to reconnect to MT5 terminal")
            print("Reconnected to MT5 terminal")


if __name__ == "__main__":
    bot = Scalper(risk_percent=0.5, risk_reward=1.5)
    try:
        bot.run()
    finally:
        bot.performance_report()




# Create performance dashboard (real-time monitoring)
# Consider using a config file for easier adjustments

"""
Add later:
self.config.update({
    'jpy_volatility_threshold': 0.015,
    'eur_volatility_threshold': 0.01,
    'high_volatility_atr': 0.003,
    'volatility_adjustment_factor': 0.7
})
"""