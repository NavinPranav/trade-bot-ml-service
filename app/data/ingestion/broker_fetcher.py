"""Options chain placeholder — production data should come from the upstream backend service."""
import pandas as pd
from loguru import logger


def fetch_options_chain(expiry: str = "current") -> pd.DataFrame:
    """
    Sensex options chain with strike, premium, OI, IV, Greeks.
    Not used by the gRPC prediction path; wire in only if the backend exposes options data to this service.
    """
    logger.warning("Options chain fetcher: placeholder. Integrate via backend when needed.")
    return pd.DataFrame(columns=[
        "strike", "expiry", "option_type", "ltp", "oi",
        "volume", "iv", "delta", "gamma", "theta", "vega"
    ])