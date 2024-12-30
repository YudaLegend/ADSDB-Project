from currency_converter import CurrencyConverter
import json
import ast
import duckdb

def read_table(con, table_name):
    return con.execute(f"SELECT * FROM {table_name};").df()


def reload_table(con, table_name, df):
    con.execute(f"DROP TABLE IF EXISTS {table_name};")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
c = CurrencyConverter()

def currency_conversion(price, from_currency):
    price_converted = c.convert(price, from_currency, 'EUR')/100
    return round(price_converted,2)


def convert_price(valor,name):
    try:
        price_dict = json.loads(valor)  
        if 'totalPrice' in price_dict:
            totalPrice = price_dict['totalPrice']
            if name in totalPrice:
                currencyCode = totalPrice['currencyCode']
                return currency_conversion(int(totalPrice[name]), currencyCode)       
        return ''
    except Exception as e:
        print(e)
        return ''

def epic_games_currency_conversion(con):
    df = read_table(con, 'epic_games')

    df["originalPrice"] = df["price"].apply(lambda x:convert_price(valor=x,name="originalPrice"))
    df["discountPrice"] = df["price"].apply(lambda x:convert_price(valor=x,name="discountPrice"))
    # Drop the price column since we just extracted the prices from it and we don't need it anymore
    df.drop(columns=["price"], inplace=True)

    reload_table(con, 'epic_games', df)


def steam_app_details_currency_conversion(con):
    df = read_table(con, 'steam_app_details')

    initial_price = []
    discount_price = []

    ## Put all currencies that don't exist in the currency_converter library to EUR conversion
    currencyToEUR = {'COP': 0.00022, 'CLP': 0.00098, 'VND': 0.000037, 'PEN': 0.25, 'AED': 0.25, 'UAH': 0.022, 'SAR': 0.25, 'KZT': 0.0019}
    for i in df['price_overview']:
        if i != '':
            try:
                price_dict = ast.literal_eval(i)
                
                from_currency = price_dict['currency']
                initial = price_dict['initial']
                discount = price_dict['final']
                if from_currency in currencyToEUR:
                    initialPrice = int(initial)/100*currencyToEUR[from_currency]
                    discountPrice = int(discount)/100*currencyToEUR[from_currency]
                    initial_price.append(round(initialPrice,2))
                    discount_price.append(round(discountPrice,2))
                else:
                    initial_price.append(currency_conversion(int(initial), from_currency))
                    discount_price.append(currency_conversion(int(discount), from_currency))
            except Exception as e:
                print(e)
                initial_price.append(0)
                discount_price.append(0)
        else:
            initial_price.append(0)
            discount_price.append(0)

    df['initial_price'] = initial_price
    df['final_price'] = discount_price
    # Drop the price_overview column since we already extracted the initial and final prices.
    df = df.drop(columns=["price_overview"])

    reload_table(con, 'steam_app_details', df)


def steam_spy_currency_conversion(con):
    df = read_table(con, 'steam_spy')
    df['price'] = df['price'].apply(lambda x: currency_conversion(float(x), 'USD'))
    df['initialprice'] = df['initialprice'].apply(lambda x: currency_conversion(float(x), 'USD'))
    reload_table(con, 'steam_spy', df)



def data_currency_conversion(con):
    try:
        epic_games_currency_conversion(con)
    except Exception as e:
        print(f"Epic Games currency conversion failed: {e}")
        return False
    
    try:
        steam_app_details_currency_conversion(con)
    except Exception as e:
        print(f"Steam App Details currency conversion failed: {e}")
        return False
    
    try:
        steam_spy_currency_conversion(con)
    except Exception as e:
        print(f"Steam Spy currency conversion failed: {e}")
        return False
        
    return True