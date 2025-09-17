# NIFTY Options Signal System

An advanced automated signal generation system for NIFTY index options trading. This system monitors market data in real-time and generates high-probability trading signals based on multiple technical indicators and market microstructure analysis.

## Features

- **Real-time Market Monitoring**: Continuous monitoring during market hours (9:15 AM - 4:00 PM IST)
- **Multi-factor Signal Generation**: Combines VWAP, RSI, SuperTrend, and market regime analysis
- **Beautiful Terminal Display**: Color-coded, real-time market status with tabulated data
- **Risk Management**: Built-in risk warnings and signal strength scoring
- **Discord Notifications**: Instant alerts to Discord when signals are generated
- **GitHub Actions Integration**: Automated daily monitoring with CI/CD
- **Comprehensive Logging**: Detailed logs for analysis and backtesting

## Signal Generation Logic

The system generates signals based on:
- VWAP deviation analysis
- RSI momentum indicators
- SuperTrend direction
- Market regime classification (Trending/Ranging/Volatile)
- Order flow and market microstructure
- Risk-adjusted scoring system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nifty-signals-system.git
cd nifty-signals-system
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.sample .env
# Edit .env and add your UPSTOX_ACCESS_TOKEN
```

## Configuration

Edit `.env` file with your settings:

```env
# Required
UPSTOX_ACCESS_TOKEN=your_access_token_here

# Optional
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
FORCE_RUN=false
MIN_SIGNAL_STRENGTH=65
SIGNAL_COOLDOWN=300
```

## Usage

### Local Development

Run the signal monitor:
```bash
python nifty_signals.py
```

### GitHub Actions (Automated)

1. Add secrets in GitHub repository settings:
   - `UPSTOX_ACCESS_TOKEN`: Your Upstox API token
   - `DISCORD_WEBHOOK_URL`: Discord webhook for notifications

2. The system will automatically run during market hours via GitHub Actions

### Manual Trigger

Trigger manually from GitHub Actions tab or via API:
```bash
gh workflow run signal-monitor
```

## Signal Output

When a signal is generated, you'll see:

```
🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 
=====================================
        TRADING SIGNAL DETECTED
=====================================
Signal Type    : CE (CALL)
Time          : 10:35:42
Spot Price    : ₹21,875.50
Strike Price  : ₹21,900
Signal Strength: 78% (STRONG)
Confidence    : HIGH ⭐⭐⭐
Entry Zone    : ₹98.00 - ₹102.00
Stop Loss     : ₹75.00 (-25%)
Target        : ₹110.00 (+10%)
Market Regime : TRENDING_UP

📊 Technical Summary:
   RSI: 58, VWAP Dev: +0.15%, Vol: 18.5%, Regime: TRENDING_UP

⚠️ Risk Warnings:
   • High volatility environment
```

## Project Structure

```
nifty-signals-system/
├── .github/
│   └── workflows/
│       └── signal-monitor.yml    # GitHub Actions workflow
├── logs/                         # Signal logs (auto-created)
├── .env                          # Your configuration
├── .env.sample                   # Sample configuration
├── nifty_signals.py             # Main application
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Market Hours

The system operates during NSE market hours:
- **Monday to Friday**: 9:15 AM - 4:00 PM IST
- **Closed**: Weekends and NSE holidays

Use `FORCE_RUN=true` to run outside market hours for testing.

## Signals Explanation

### Signal Types
- **CE (Call Option)**: Bullish signal expecting upward movement
- **PE (Put Option)**: Bearish signal expecting downward movement

### Signal Strength
- **85%+**: Very Strong signal
- **75-84%**: Strong signal
- **65-74%**: Moderate signal
- **Below 65%**: No signal generated

### Risk Management
- **Stop Loss**: 25% of entry premium
- **Target**: 10% of entry premium
- **Risk-Reward Ratio**: 1:2.5

## Troubleshooting

### Common Issues

1. **"UPSTOX_ACCESS_TOKEN not set"**
   - Edit `.env` file and add your token
   - Get token from: https://account.upstox.com/developer/apps

2. **"Failed to connect"**
   - Check internet connection
   - Verify token is valid and not expired
   - Ensure market is open

3. **No signals generated**
   - System requires minimum 100 ticks before first signal
   - Signals have 5-minute cooldown period
   - Check if market conditions meet signal criteria

### Debug Mode

Enable debug mode for detailed output:
```env
LOG_LEVEL=DEBUG
DEBUG_MODE=true
```

## Disclaimer

This system is for educational and informational purposes only. It generates trading signals but does not execute actual trades. Always perform your own analysis and risk assessment before making any trading decisions. Trading in options involves substantial risk and is not suitable for all investors.

## Support

- Create an issue for bug reports
- Check logs in `logs/` directory for troubleshooting
- Join our Discord for community support

## License

MIT License - See LICENSE file for details
