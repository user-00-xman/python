CURRENCIES = {
    'USD': 1,
    'EUR': 1.06,
    'YEN': 0.0067,
    'GBP': 1.23,
    'AUD': 0.64,
    'CAD': 0.74
}

def to_usd(currency_code, amount):
    """
    Convert the given amount from the specified currency to USD.
    """
    if currency_code not in CURRENCIES:
        raise ValueError(f"{currency_code} is not supported")
    if amount < 0:
        raise Exception("Invalid amount")

    # Get the conversion rate for the specified currency
    conversion_rate = CURRENCIES[currency_code]

    # Calculate the equivalent in USD
    usd_value = amount * conversion_rate

    return usd_value

def from_usd(currency_code, amount):
    """
    Convert the given amount from USD to the specified currency.
    """
    if currency_code not in CURRENCIES:
        raise ValueError(f"{currency_code} is not supported")
    if amount < 0:
        raise Exception("Invalid amount")

    # Get the conversion rate for the specified currency
    conversion_rate = CURRENCIES[currency_code]

    # Calculate the equivalent in the target currency
    currency_value = amount / conversion_rate

    # Round the result to 4 floating digits
    return round(currency_value, 4)

def convert(from_currency, amount, to_currency):
    try:
        # Convert from the original currency to usd
        amount_in_usd = to_usd(from_currency, amount)

        equivalent_amount = from_usd(to_currency, amount_in_usd)

        print(f"{amount} {from_currency} is {equivalent_amount} {to_currency}")
    
    except Exception as e:
        print(e)



