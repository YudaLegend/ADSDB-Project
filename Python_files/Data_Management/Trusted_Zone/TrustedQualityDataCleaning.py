import duckdb

def read_table(con, table_name):
    return con.execute(f"SELECT * FROM {table_name};").df()

def reload_table(con, table_name, df):
    con.execute(f"DROP TABLE IF EXISTS {table_name};")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

def epic_games_data_cleaning(con):
    df = read_table(con, 'epic_games')
    df = df.drop_duplicates(subset=['id'])
    df.drop(columns=["namespace", "id", "url", "urlSlug", "keyImages", "items", "tags","productSlug"], inplace=True)
    reload_table(con, 'epic_games', df)

def steam_current_players_cleaning(con):
    df = read_table(con, 'steam_current_players')
    df.drop_duplicates(subset=['app_id'], inplace=True)
    df['app_id'] = df['app_id'].apply(lambda x: int(x))
    df = df.drop(columns=["result"])
    reload_table(con, 'steam_current_players', df)

def steam_app_details_cleaning(con):
    df = read_table(con, 'steam_app_details')
    df.drop_duplicates(subset=['steam_appid'], inplace=True)
    df.drop(columns=['controller_support', 'header_image', 'capsule_image', 'capsule_imagev5', 'website', 'legal_notice', 'pc_requirements',
                        'mac_requirements', 'linux_requirements', 'screenshots', 'movies', 'ext_user_account_notice', 'drm_notice', 
                        'support_info', 'background', 'background_raw', 'reviews', 'content_descriptors', 'ratings', 
                        'demos', 'packages','package_groups',"fullgame"], inplace=True)
    reload_table(con, 'steam_app_details', df)

def steam_spy_cleaning(con):
    df = read_table(con, 'steam_spy')
    df.drop_duplicates(subset=['appid'], inplace=True)
    df = df.drop(columns=["score_rank","tags"])
    reload_table(con, 'steam_spy', df)


def data_cleaning(con):
    try:
        epic_games_data_cleaning(con)
    except Exception as e:
        print(f"Error in epic_games_data_cleaning: {e}")
        return False
    
    try:
        steam_app_details_cleaning(con)
    except Exception as e:
        print(f"Error in steam_app_details_cleaning: {e}")
        return False
    
    try:
        steam_spy_cleaning(con)
    except Exception as e:
        print(f"Error in steam_spy_cleaning: {e}")
        return False
    
    try:
        steam_current_players_cleaning(con)
    except Exception as e:
        print(f"Error in steam_current_players_cleaning: {e}")
        return False
    return True

