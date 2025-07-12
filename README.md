# MT5 Scalping Trading Bot


A production-ready scalping bot for MetaTrader 5 (MT5) designed for EURUSD and USDJPY pairs with advanced risk management and trade execution features.

## Features

- **Multi-timeframe entry signals** (M1/M5 analysis)
- **ATR-based position sizing** with volatility filters
- **Advanced risk management**:
  - Daily drawdown limits
  - Position count limits
  - Dynamic risk adjustment
- **Order flow analysis** (market depth integration)
- **Professional trade management**:
  - Trailing stops with ATR adaptation
  - Partial profit taking
  - Time-based exits
- **Performance analytics** with detailed reporting
- **Broker synchronization** with time drift detection

## Requirements

- MetaTrader 5 terminal installed
- Python 3.7+
- MT5 Python package (`pip install MetaTrader5`)
- Pandas (`pip install pandas`)
- NumPy (`pip install numpy`)
- Pytz (`pip install pytz`)





Fixes Needed:

    Remove duplicate order_send call

    Define missing symbol variable in run()

    Initialize last_dst_state

    Implement _handle_partial_fill()

    Implement is_acceptable_volatility()
