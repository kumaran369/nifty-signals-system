#!/usr/bin/env python3
"""
NIFTY Option Scalping Signal System v3.5
Advanced Signal Generation with Beautiful Terminal Output
Continuous Market Monitoring (9:15 AM - 4:00 PM IST)
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import signal as sig

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats
import requests
import pytz
from colorama import init, Fore, Back, Style
from tabulate import tabulate
from dotenv import load_dotenv

# Upstox imports
try:
    import upstox_client
    from upstox_client.rest import ApiException
except ImportError:
    print("Error: upstox-python-sdk not installed")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Initialize colorama for Windows support
init(autoreset=True)

# Load environment variables
load_dotenv()

# ================ CONFIGURATION ================

class Config:
    """Configuration management from environment"""
    
    # API Configuration
    ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "")
    DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "")
    
    # Trading Parameters
    FORCE_RUN = os.getenv("FORCE_RUN", "false").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"
    COLORFUL_LOGS = os.getenv("COLORFUL_LOGS", "true").lower() == "true"
    
    # Signal Parameters
    MIN_SIGNAL_STRENGTH = int(os.getenv("MIN_SIGNAL_STRENGTH", "65"))
    SIGNAL_COOLDOWN = int(os.getenv("SIGNAL_COOLDOWN", "300"))
    MIN_TICK_WARMUP = int(os.getenv("MIN_TICK_WARMUP", "100"))
    MAX_SIGNALS_PER_DAY = int(os.getenv("MAX_SIGNALS_PER_DAY", "10"))
    
    # Technical Indicators
    RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
    VWAP_SENSITIVITY = float(os.getenv("VWAP_SENSITIVITY", "0.001"))
    VOLATILITY_WINDOW = int(os.getenv("VOLATILITY_WINDOW", "20"))
    SUPERTREND_PERIOD = int(os.getenv("SUPERTREND_PERIOD", "10"))
    SUPERTREND_MULTIPLIER = float(os.getenv("SUPERTREND_MULTIPLIER", "3.0"))
    
    # Risk Parameters
    MAX_VOLATILITY = float(os.getenv("MAX_VOLATILITY", "0.35"))
    MIN_LIQUIDITY_SCORE = int(os.getenv("MIN_LIQUIDITY_SCORE", "60"))
    
    # Market Hours (IST)
    MARKET_OPEN_HOUR = int(os.getenv("MARKET_OPEN_HOUR", "9"))
    MARKET_OPEN_MINUTE = int(os.getenv("MARKET_OPEN_MINUTE", "15"))
    MARKET_CLOSE_HOUR = int(os.getenv("MARKET_CLOSE_HOUR", "16"))  # Extended to 4 PM
    MARKET_CLOSE_MINUTE = int(os.getenv("MARKET_CLOSE_MINUTE", "0"))
    
    # Display Settings
    UPDATE_FREQUENCY = float(os.getenv("UPDATE_FREQUENCY", "0.5"))
    CLEAR_SCREEN = os.getenv("CLEAR_SCREEN", "false").lower() == "true"
    SHOW_INDICATORS = os.getenv("SHOW_INDICATORS", "true").lower() == "true"
    
    # Alert Settings
    SOUND_ALERTS = os.getenv("SOUND_ALERTS", "false").lower() == "true"
    DESKTOP_NOTIFICATIONS = os.getenv("DESKTOP_NOTIFICATIONS", "false").lower() == "true"
    
    # Debug Settings
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
    SAVE_TICK_DATA = os.getenv("SAVE_TICK_DATA", "false").lower() == "true"
    
    # Timezone
    IST = pytz.timezone('Asia/Kolkata')

# ================ ENUMS AND DATACLASSES ================

class MarketRegime(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    QUIET = "QUIET"

class SignalStrength(Enum):
    VERY_STRONG = "VERY_STRONG"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"

@dataclass
class MarketMetrics:
    """Real-time market metrics"""
    price: float = 0.0
    change: float = 0.0
    change_pct: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    volume: int = 0
    vwap: float = 0.0
    rsi: float = 50.0
    volatility: float = 0.0
    spread: float = 0.0
    tick_count: int = 0
    last_update: datetime = field(default_factory=datetime.now)

@dataclass
class SignalAlert:
    """Signal alert data"""
    timestamp: datetime
    signal_type: str  # CE/PE
    spot_price: float
    strike: int
    strength: float
    confidence: str
    entry_zone: Tuple[float, float]
    stop_loss: float
    target: float
    risk_warnings: List[str]
    technical_summary: str
    regime: str

# ================ TERMINAL DISPLAY ================

class TerminalDisplay:
    """Beautiful terminal output management"""
    
    def __init__(self):
        self.last_update = time.time()
        self.market_metrics = MarketMetrics()
        self.signals_today = []
        self.active_monitoring = True
        self.status_line = ""
        
        # Colors
        self.colors = {
            'header': Fore.CYAN + Style.BRIGHT,
            'success': Fore.GREEN + Style.BRIGHT,
            'warning': Fore.YELLOW + Style.BRIGHT,
            'danger': Fore.RED + Style.BRIGHT,
            'info': Fore.BLUE + Style.BRIGHT,
            'normal': Style.RESET_ALL
        }
    
    def clear_screen(self):
        """Clear terminal screen"""
        if Config.CLEAR_SCREEN:
            os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_header(self):
        """Display system header"""
        self.clear_screen()
        print(self.colors['header'] + "="*80)
        print(self.colors['header'] + "NIFTY OPTIONS SIGNAL SYSTEM v3.5".center(80))
        print(self.colors['header'] + f"Market Hours: 9:15 AM - 4:00 PM IST".center(80))
        print(self.colors['header'] + "="*80 + self.colors['normal'])
    
    def display_market_status(self, metrics: MarketMetrics, regime: MarketRegime):
        """Display real-time market status"""
        now = datetime.now(Config.IST)
        
        # Prepare color based on change
        change_color = self.colors['success'] if metrics.change >= 0 else self.colors['danger']
        
        # Market status box
        status_lines = [
            ["Market Status", f"{now.strftime('%H:%M:%S')}"],
            ["", ""],
            ["NIFTY", f"{metrics.price:,.2f}"],
            ["Change", change_color + f"{metrics.change:+.2f} ({metrics.change_pct:+.2f}%)" + self.colors['normal']],
            ["Day Range", f"{metrics.day_low:,.2f} - {metrics.day_high:,.2f}"],
            ["Volume", f"{metrics.volume:,}"],
            ["", ""],
            ["VWAP", f"{metrics.vwap:,.2f}"],
            ["RSI", self._format_rsi(metrics.rsi)],
            ["Volatility", f"{metrics.volatility:.2%}"],
            ["Regime", self._format_regime(regime)],
            ["", ""],
            ["Signals Today", f"{len(self.signals_today)}"],
            ["Tick Count", f"{metrics.tick_count:,}"]
        ]
        
        # Create table
        table = tabulate(status_lines, tablefmt="fancy_grid")
        
        # Print with proper spacing
        print("\r" + " "*100, end='\r')  # Clear line
        print(f"\n{table}\n")
        
        # Show active signals
        if self.signals_today:
            self._display_recent_signals()
    
    def _format_rsi(self, rsi: float) -> str:
        """Format RSI with color coding"""
        if rsi > 70:
            return self.colors['danger'] + f"{rsi:.0f} (Overbought)" + self.colors['normal']
        elif rsi < 30:
            return self.colors['success'] + f"{rsi:.0f} (Oversold)" + self.colors['normal']
        elif rsi > 55:
            return self.colors['success'] + f"{rsi:.0f} (Bullish)" + self.colors['normal']
        elif rsi < 45:
            return self.colors['danger'] + f"{rsi:.0f} (Bearish)" + self.colors['normal']
        else:
            return f"{rsi:.0f} (Neutral)"
    
    def _format_regime(self, regime: MarketRegime) -> str:
        """Format regime with emoji and color"""
        regime_map = {
            MarketRegime.TRENDING_UP: self.colors['success'] + "📈 Trending Up" + self.colors['normal'],
            MarketRegime.TRENDING_DOWN: self.colors['danger'] + "📉 Trending Down" + self.colors['normal'],
            MarketRegime.RANGING: "↔️  Ranging",
            MarketRegime.VOLATILE: self.colors['warning'] + "🌊 Volatile" + self.colors['normal'],
            MarketRegime.QUIET: "😴 Quiet"
        }
        return regime_map.get(regime, str(regime.value))
    
    def display_signal_alert(self, signal: SignalAlert):
        """Display signal alert with beautiful formatting"""
        print("\n" + self.colors['warning'] + "🚨 "*20 + self.colors['normal'])
        print(self.colors['header'] + "="*80)
        print(self.colors['header'] + f"{'TRADING SIGNAL DETECTED':^80}")
        print(self.colors['header'] + "="*80 + self.colors['normal'])
        
        # Signal type with color
        signal_color = self.colors['success'] if signal.signal_type == 'CE' else self.colors['danger']
        
        # Signal details table
        details = [
            ["Signal Type", signal_color + f"{signal.signal_type} (CALL)" if signal.signal_type == 'CE' else f"{signal.signal_type} (PUT)" + self.colors['normal']],
            ["Time", signal.timestamp.strftime("%H:%M:%S")],
            ["Spot Price", f"₹{signal.spot_price:,.2f}"],
            ["Strike Price", f"₹{signal.strike:,}"],
            ["Signal Strength", self._format_strength(signal.strength)],
            ["Confidence", self._format_confidence(signal.confidence)],
            ["Entry Zone", f"₹{signal.entry_zone[0]:.2f} - ₹{signal.entry_zone[1]:.2f}"],
            ["Stop Loss", self.colors['danger'] + f"₹{signal.stop_loss:.2f} (-25%)" + self.colors['normal']],
            ["Target", self.colors['success'] + f"₹{signal.target:.2f} (+10%)" + self.colors['normal']],
            ["Market Regime", self._format_regime(MarketRegime[signal.regime])]
        ]
        
        print(tabulate(details, tablefmt="fancy_grid"))
        
        # Technical summary
        print(f"\n{self.colors['info']}📊 Technical Summary:{self.colors['normal']}")
        print(f"   {signal.technical_summary}")
        
        # Risk warnings
        if signal.risk_warnings:
            print(f"\n{self.colors['warning']}⚠️  Risk Warnings:{self.colors['normal']}")
            for warning in signal.risk_warnings:
                print(f"   • {warning}")
        
        print("\n" + self.colors['warning'] + "🚨 "*20 + self.colors['normal'] + "\n")
        
        # Add to today's signals
        self.signals_today.append(signal)
        
        # Sound alert if enabled
        if Config.SOUND_ALERTS:
            self._play_alert_sound()
    
    def _format_strength(self, strength: float) -> str:
        """Format signal strength with color"""
        if strength >= 85:
            return self.colors['success'] + f"{strength:.0f}% (VERY STRONG)" + self.colors['normal']
        elif strength >= 75:
            return self.colors['success'] + f"{strength:.0f}% (STRONG)" + self.colors['normal']
        elif strength >= 65:
            return self.colors['warning'] + f"{strength:.0f}% (MODERATE)" + self.colors['normal']
        else:
            return f"{strength:.0f}% (WEAK)"
    
    def _format_confidence(self, confidence: str) -> str:
        """Format confidence level with color"""
        conf_map = {
            "HIGH": self.colors['success'] + "HIGH ⭐⭐⭐" + self.colors['normal'],
            "MEDIUM": self.colors['warning'] + "MEDIUM ⭐⭐" + self.colors['normal'],
            "LOW": "LOW ⭐"
        }
        return conf_map.get(confidence, confidence)
    
    def _display_recent_signals(self):
        """Display recent signals summary"""
        if not self.signals_today:
            return
        
        print(f"\n{self.colors['info']}📋 Today's Signals:{self.colors['normal']}")
        
        recent_signals = []
        for sig in self.signals_today[-5:]:  # Last 5 signals
            recent_signals.append([
                sig.timestamp.strftime("%H:%M"),
                sig.signal_type,
                f"{sig.strike}",
                f"{sig.strength:.0f}%",
                sig.confidence
            ])
        
        headers = ["Time", "Type", "Strike", "Strength", "Confidence"]
        print(tabulate(recent_signals, headers=headers, tablefmt="simple"))
    
    def display_status_line(self, message: str):
        """Display status line without newline"""
        print(f"\r{message:<100}", end='', flush=True)
    
    def _play_alert_sound(self):
        """Play alert sound (cross-platform)"""
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(1000, 500)
            else:
                os.system('echo -n "\a"')
        except:
            pass

# ================ SIGNAL GENERATION SYSTEM ================

class SignalGenerator:
    """Core signal generation logic"""
    
    def __init__(self, display: TerminalDisplay):
        self.display = display
        self.price_buffer = deque(maxlen=1000)
        self.volume_buffer = deque(maxlen=1000)
        self.tick_count = 0
        self.last_signal_time = 0
        self.daily_signal_count = 0
        
        # Technical indicator states
        self.vwap_state = {'sum_pv': 0, 'sum_v': 0, 'day_start': None}
        self.rsi_state = {'gains': deque(maxlen=Config.RSI_PERIOD), 
                         'losses': deque(maxlen=Config.RSI_PERIOD)}
        self.prev_price = None
        
        # Market open price
        self.open_price = None
        self.day_high = 0
        self.day_low = float('inf')
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def process_tick(self, tick_data: Dict) -> Optional[SignalAlert]:
        """Process market tick and generate signals"""
        try:
            # Extract tick data
            price = float(tick_data.get('ltp', 0))
            volume = float(tick_data.get('volume', tick_data.get('ltq', 1)))
            
            if not self._validate_tick(price, volume):
                return None
            
            # Update buffers
            self.tick_count += 1
            self.price_buffer.append(price)
            self.volume_buffer.append(volume)
            
            # Update day high/low
            if self.open_price is None:
                self.open_price = price
            self.day_high = max(self.day_high, price)
            self.day_low = min(self.day_low, price)
            
            # Calculate change
            change = price - self.open_price if self.open_price else 0
            change_pct = (change / self.open_price * 100) if self.open_price else 0
            
            # Calculate indicators
            vwap = self._calculate_vwap(price, volume)
            rsi = self._calculate_rsi(price)
            volatility = self._calculate_volatility()
            regime = self._detect_regime()
            
            # Update display metrics
            self.display.market_metrics = MarketMetrics(
                price=price,
                change=change,
                change_pct=change_pct,
                day_high=self.day_high,
                day_low=self.day_low,
                volume=int(self.vwap_state['sum_v']),
                vwap=vwap,
                rsi=rsi,
                volatility=volatility,
                spread=self._estimate_spread(price),
                tick_count=self.tick_count,
                last_update=datetime.now(Config.IST)
            )
            
            # Update display
            if time.time() - self.display.last_update > Config.UPDATE_FREQUENCY:
                self.display.display_market_status(self.display.market_metrics, regime)
                self.display.last_update = time.time()
            
            # Check for signals
            if self._should_generate_signal():
                signal = self._evaluate_signal_conditions(price, vwap, rsi, volatility, regime)
                if signal:
                    self.last_signal_time = time.time()
                    self.daily_signal_count += 1
                    return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing tick: {e}")
            return None
    
    def _validate_tick(self, price: float, volume: float) -> bool:
        """Validate tick data"""
        if price <= 0 or price > 50000:
            return False
        if volume < 0:
            return False
        if self.prev_price and abs(price - self.prev_price) / self.prev_price > 0.02:
            return False
        return True
    
    def _calculate_vwap(self, price: float, volume: float) -> float:
        """Calculate VWAP"""
        current_date = datetime.now().date()
        
        if self.vwap_state['day_start'] != current_date:
            self.vwap_state = {
                'sum_pv': price * volume,
                'sum_v': volume,
                'day_start': current_date
            }
        else:
            self.vwap_state['sum_pv'] += price * volume
            self.vwap_state['sum_v'] += volume
        
        if self.vwap_state['sum_v'] > 0:
            return self.vwap_state['sum_pv'] / self.vwap_state['sum_v']
        return price
    
    def _calculate_rsi(self, price: float) -> float:
        """Calculate RSI"""
        if self.prev_price is None:
            self.prev_price = price
            return 50.0
        
        change = price - self.prev_price
        gain = max(0, change)
        loss = max(0, -change)
        
        self.rsi_state['gains'].append(gain)
        self.rsi_state['losses'].append(loss)
        
        if len(self.rsi_state['gains']) < Config.RSI_PERIOD:
            self.prev_price = price
            return 50.0
        
        avg_gain = sum(self.rsi_state['gains']) / Config.RSI_PERIOD
        avg_loss = sum(self.rsi_state['losses']) / Config.RSI_PERIOD
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        self.prev_price = price
        return rsi
    
    def _calculate_volatility(self) -> float:
        """Calculate realized volatility"""
        if len(self.price_buffer) < Config.VOLATILITY_WINDOW:
            return 0.15
        
        prices = list(self.price_buffer)[-Config.VOLATILITY_WINDOW:]
        returns = np.diff(np.log(prices))
        
        if len(returns) > 0:
            return np.std(returns) * np.sqrt(252 * 375)  # Annualized
        return 0.15
    
    def _estimate_spread(self, price: float) -> float:
        """Estimate bid-ask spread"""
        # Simple estimation based on price level
        if price < 20000:
            return 2.0
        elif price < 25000:
            return 3.0
        else:
            return 5.0
    
    def _detect_regime(self) -> MarketRegime:
        """Detect market regime"""
        if len(self.price_buffer) < 50:
            return MarketRegime.QUIET
        
        prices = list(self.price_buffer)[-50:]
        
        # Calculate trend
        x = np.arange(len(prices))
        slope, _, r_squared, _, _ = stats.linregress(x, prices)
        trend_strength = abs(slope) / np.mean(prices)
        
        # Calculate volatility
        volatility = self._calculate_volatility()
        
        # Classify regime
        if r_squared > 0.7 and trend_strength > 0.0001:
            if slope > 0:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        elif volatility > 0.25:
            return MarketRegime.VOLATILE
        elif volatility < 0.10:
            return MarketRegime.QUIET
        else:
            return MarketRegime.RANGING
    
    def _should_generate_signal(self) -> bool:
        """Check if we should evaluate for signals"""
        # Warmup check
        if self.tick_count < Config.MIN_TICK_WARMUP:
            return False
        
        # Daily limit check
        if self.daily_signal_count >= Config.MAX_SIGNALS_PER_DAY:
            return False
        
        # Cooldown check
        if time.time() - self.last_signal_time < Config.SIGNAL_COOLDOWN:
            return False
        
        return True
    
    def _evaluate_signal_conditions(self, price: float, vwap: float, rsi: float, 
                                  volatility: float, regime: MarketRegime) -> Optional[SignalAlert]:
        """Evaluate market conditions for signal generation"""
        
        # Calculate signal scores
        vwap_deviation = (price - vwap) / vwap if vwap > 0 else 0
        
        # Technical score
        tech_score = 0
        
        # VWAP component
        if abs(vwap_deviation) > Config.VWAP_SENSITIVITY:
            tech_score += 30 * np.sign(vwap_deviation)
        
        # RSI component
        if 40 < rsi < 60:
            tech_score += (rsi - 50) * 2
        elif rsi > 70:
            tech_score -= 20
        elif rsi < 30:
            tech_score += 20
        
        # Regime component
        regime_scores = {
            MarketRegime.TRENDING_UP: 20,
            MarketRegime.TRENDING_DOWN: -20,
            MarketRegime.RANGING: 0,
            MarketRegime.VOLATILE: -10,
            MarketRegime.QUIET: 5
        }
        tech_score += regime_scores.get(regime, 0)
        
        # Momentum component
        if len(self.price_buffer) > 10:
            momentum = (price - self.price_buffer[-10]) / self.price_buffer[-10] * 100
            if abs(momentum) > 0.1:
                tech_score += np.sign(momentum) * 15
        
        # Risk adjustments
        risk_score = 100
        if volatility > Config.MAX_VOLATILITY:
            risk_score -= 30
        
        # Time of day adjustment
        hour = datetime.now().hour
        if hour == 9 or hour == 15:
            risk_score -= 20
        
        # Calculate final score
        final_score = abs(tech_score) * (risk_score / 100)
        
        # Generate signal if score is sufficient
        if final_score >= Config.MIN_SIGNAL_STRENGTH:
            signal_type = "CE" if tech_score > 0 else "PE"
            
            # Calculate strike and levels
            strike = round(price / 50) * 50
            if signal_type == "CE" and price > strike:
                strike += 50
            elif signal_type == "PE" and price < strike:
                strike -= 50
            
            # Entry, stop, target (hypothetical values)
            entry_premium = 100  # Placeholder
            stop_loss = entry_premium * 0.75
            target = entry_premium * 1.10
            
            # Risk warnings
            warnings = []
            if volatility > 0.25:
                warnings.append("High volatility environment")
            if hour in [9, 15]:
                warnings.append("Opening/closing hour volatility")
            if abs(vwap_deviation) > 0.005:
                warnings.append("Price extended from VWAP")
            
            # Technical summary
            tech_summary = (
                f"RSI: {rsi:.0f}, "
                f"VWAP Dev: {vwap_deviation*100:+.2f}%, "
                f"Vol: {volatility:.2%}, "
                f"Regime: {regime.value}"
            )
            
            # Confidence level
            if final_score >= 80:
                confidence = "HIGH"
            elif final_score >= 70:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            
            return SignalAlert(
                timestamp=datetime.now(Config.IST),
                signal_type=signal_type,
                spot_price=price,
                strike=strike,
                strength=final_score,
                confidence=confidence,
                entry_zone=(entry_premium * 0.98, entry_premium * 1.02),
                stop_loss=stop_loss,
                target=target,
                risk_warnings=warnings,
                technical_summary=tech_summary,
                regime=regime.name
            )
        
        return None

# ================ MAIN APPLICATION ================

class NiftySignalSystem:
    """Main application coordinator"""
    
    def __init__(self):
        self.display = TerminalDisplay()
        self.signal_generator = SignalGenerator(self.display)
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
        # Setup logging
        self._setup_logging()
        
        # Signal handlers
        sig.signal(sig.SIGINT, self._signal_handler)
        sig.signal(sig.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        if Config.LOG_TO_FILE:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(
                log_dir, 
                f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            
            handlers = [logging.FileHandler(log_file)]
        else:
            handlers = []
        
        if not Config.COLORFUL_LOGS or Config.DEBUG_MODE:
            handlers.append(logging.StreamHandler(sys.stdout))
        
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n\n{self.display.colors['warning']}Shutting down gracefully...{self.display.colors['normal']}")
        self.is_running = False
        sys.exit(0)
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        if not Config.ACCESS_TOKEN:
            print(f"{self.display.colors['danger']}❌ Error: UPSTOX_ACCESS_TOKEN not set in .env file{self.display.colors['normal']}")
            return False
        
        if len(Config.ACCESS_TOKEN) < 20:
            print(f"{self.display.colors['danger']}❌ Error: Invalid UPSTOX_ACCESS_TOKEN{self.display.colors['normal']}")
            return False
        
        return True
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        if Config.FORCE_RUN:
            return True
        
        now = datetime.now(Config.IST)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        market_open = now.replace(
            hour=Config.MARKET_OPEN_HOUR,
            minute=Config.MARKET_OPEN_MINUTE,
            second=0
        )
        market_close = now.replace(
            hour=Config.MARKET_CLOSE_HOUR,
            minute=Config.MARKET_CLOSE_MINUTE,
            second=0
        )
        
        return market_open <= now <= market_close
    
    def connect_to_market(self):
        """Connect to market data stream"""
        try:
            configuration = upstox_client.Configuration()
            configuration.access_token = Config.ACCESS_TOKEN
            api_client = upstox_client.ApiClient(configuration)
            
            def on_message(message):
                """Handle incoming market data"""
                try:
                    if "feeds" in message:
                        for inst_key, feed_data in message["feeds"].items():
                            if "NSE_INDEX|Nifty 50" in inst_key:
                                ltpc = feed_data.get("ltpc", {})
                                if ltpc and ltpc.get('ltp'):
                                    tick_data = {
                                        'ltp': ltpc.get('ltp', 0),
                                        'volume': ltpc.get('ltq', 1)
                                    }
                                    
                                    # Process tick and check for signals
                                    signal = self.signal_generator.process_tick(tick_data)
                                    if signal:
                                        self.display.display_signal_alert(signal)
                                        self._send_discord_alert(signal)
                                        
                except Exception as e:
                    logging.error(f"Message processing error: {e}")
            
            def on_error(error):
                """Handle connection errors"""
                logging.error(f"WebSocket error: {error}")
                if self.is_running:
                    self._handle_reconnect()
            
            def on_close():
                """Handle connection close"""
                logging.info("WebSocket connection closed")
                if self.is_running:
                    self._handle_reconnect()
            
            # Create streamer
            self.display.display_status_line("📡 Connecting to market data stream...")
            
            streamer = upstox_client.MarketDataStreamerV3(
                api_client,
                ["NSE_INDEX|Nifty 50"],
                "full"
            )
            
            # Set event handlers
            streamer.on("message", on_message)
            streamer.on("error", on_error)
            streamer.on("close", on_close)
            
            # Connect
            streamer.connect()
            self.reconnect_attempts = 0
            
            print(f"\n{self.display.colors['success']}✅ Connected to market data stream{self.display.colors['normal']}\n")
            
            # Keep connection alive
            while self.is_running:
                time.sleep(1)
                
        except Exception as e:
            logging.error(f"Connection error: {e}")
            print(f"\n{self.display.colors['danger']}❌ Failed to connect: {e}{self.display.colors['normal']}")
            
            if self.is_running:
                self._handle_reconnect()
    
    def _handle_reconnect(self):
        """Handle reconnection with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print(f"\n{self.display.colors['danger']}❌ Max reconnection attempts reached. Exiting.{self.display.colors['normal']}")
            self.is_running = False
            return
        
        self.reconnect_attempts += 1
        wait_time = min(60, 5 * (2 ** (self.reconnect_attempts - 1)))  # Exponential backoff
        
        print(f"\n{self.display.colors['warning']}🔄 Reconnecting in {wait_time} seconds... (Attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}){self.display.colors['normal']}")
        time.sleep(wait_time)
        
        if self.is_running:
            self.connect_to_market()
    
    def _send_discord_alert(self, signal: SignalAlert):
        """Send alert to Discord webhook"""
        if not Config.DISCORD_WEBHOOK:
            return
        
        try:
            color = 0x00ff00 if signal.signal_type == "CE" else 0xff0000
            
            embed = {
                "title": f"🚨 {signal.signal_type} Signal Alert",
                "color": color,
                "timestamp": signal.timestamp.isoformat(),
                "fields": [
                    {"name": "Spot Price", "value": f"₹{signal.spot_price:,.2f}", "inline": True},
                    {"name": "Strike", "value": f"₹{signal.strike:,}", "inline": True},
                    {"name": "Strength", "value": f"{signal.strength:.0f}%", "inline": True},
                    {"name": "Entry Zone", "value": f"₹{signal.entry_zone[0]:.2f} - ₹{signal.entry_zone[1]:.2f}", "inline": True},
                    {"name": "Stop Loss", "value": f"₹{signal.stop_loss:.2f}", "inline": True},
                    {"name": "Target", "value": f"₹{signal.target:.2f}", "inline": True},
                    {"name": "Technical", "value": signal.technical_summary, "inline": False},
                    {"name": "Warnings", "value": "\n".join(signal.risk_warnings) if signal.risk_warnings else "None", "inline": False}
                ]
            }
            
            data = {"embeds": [embed]}
            requests.post(Config.DISCORD_WEBHOOK, json=data, timeout=5)
            
        except Exception as e:
            logging.error(f"Discord notification error: {e}")
    
    def run(self):
        """Main run method"""
        self.display.display_header()
        
        # Validate configuration
        if not self.validate_config():
            return
        
        # Check market hours
        if not self.is_market_open():
            print(f"\n{self.display.colors['warning']}⚠️  Market is closed{self.display.colors['normal']}")
            print(f"Market hours: {Config.MARKET_OPEN_HOUR:02d}:{Config.MARKET_OPEN_MINUTE:02d} - {Config.MARKET_CLOSE_HOUR:02d}:{Config.MARKET_CLOSE_MINUTE:02d} IST")
            
            if not Config.FORCE_RUN:
                print("\nSet FORCE_RUN=true in .env to run anyway")
                return
            else:
                print(f"\n{self.display.colors['info']}ℹ️  FORCE_RUN enabled - continuing...{self.display.colors['normal']}")
        
        self.is_running = True
        
        # Start time
        start_time = datetime.now(Config.IST)
        print(f"\n{self.display.colors['info']}📅 Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S IST')}{self.display.colors['normal']}")
        
        try:
            # Connect to market
            self.connect_to_market()
            
        except KeyboardInterrupt:
            print(f"\n{self.display.colors['warning']}⚠️  Interrupted by user{self.display.colors['normal']}")
            
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            print(f"\n{self.display.colors['danger']}❌ Unexpected error: {e}{self.display.colors['normal']}")
            
        finally:
            self.is_running = False
            
            # Summary
            end_time = datetime.now(Config.IST)
            duration = end_time - start_time
            
            print(f"\n{self.display.colors['header']}{'='*80}")
            print("SESSION SUMMARY".center(80))
            print("="*80 + self.display.colors['normal'])
            print(f"Duration: {duration}")
            print(f"Signals Generated: {len(self.display.signals_today)}")
            print(f"Ticks Processed: {self.signal_generator.tick_count:,}")
            
            if self.display.signals_today:
                print(f"\n{self.display.colors['info']}Signal Summary:{self.display.colors['normal']}")
                for i, sig in enumerate(self.display.signals_today, 1):
                    print(f"{i}. {sig.timestamp.strftime('%H:%M:%S')} - {sig.signal_type} {sig.strike} ({sig.strength:.0f}%)")
            
            print(f"\n{self.display.colors['header']}{'='*80}{self.display.colors['normal']}")

# ================ MAIN ENTRY POINT ================

def main():
    """Main entry point"""
    # Check for .env file
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("\nCopying from .env.sample...")
        
        if os.path.exists('.env.sample'):
            import shutil
            shutil.copy('.env.sample', '.env')
            print("✅ Created .env file from sample")
            print("\n⚠️  Please edit .env and add your UPSTOX_ACCESS_TOKEN")
        else:
            print("❌ .env.sample not found either!")
        
        return
    
    # Create and run system
    system = NiftySignalSystem()
    system.run()

if __name__ == "__main__":
    main()
