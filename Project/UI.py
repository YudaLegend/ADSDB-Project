import tkinter as tk

import TemporalToPersistent
import PersistentToFormatted
import FormattedToTrusted
import TrustedQuality
import TrustedToExploitation
import ExploitationQualityKPI
import duckdb

# Create the main window
root = tk.Tk()
root.title("Data Management Backbone")
root.geometry("800x500")  # Set the size of the window

start = False



def connectToDatabase(db_path):
    con = duckdb.connect(database=db_path, read_only=False)
    return con

def write_to_terminal(message):
    terminal_text.config(state=tk.NORMAL)  # Enable text widget to insert new text
    terminal_text.insert(tk.END, message + "\n")
    terminal_text.see(tk.END)  # Automatically scroll to the bottom
    terminal_text.config(state=tk.DISABLED)  # Disable editing again
    root.update()


def temporalToPersistent():
    root.after(1000)
    TemporalToPersistent.create_folders()
    write_to_terminal("All the zone folders created !!!")

    root.after(1000)
    TemporalToPersistent.move_files()
    write_to_terminal("Completed the file organization !!!")


def persistentToFormatted():
    root.after(1000)
    con = connectToDatabase('./Project/Formatted Zone/formatted_zone.duckdb')
    write_to_terminal("Connected to the database !!!")

    root.after(500)
    PersistentToFormatted.epicGamesDataset(con)
    write_to_terminal("Completed the Epic Games dataset formatting !!!")

    root.after(1000)
    PersistentToFormatted.steamSpyDataset(con)
    write_to_terminal("Completed the SteamSpy dataset formatting !!!")

    root.after(1000)
    PersistentToFormatted.steamCurrentPlayerDataset(con)
    write_to_terminal("Completed the Steam current player dataset formatting !!!")

    root.after(1000)
    PersistentToFormatted.steamGameDetailsDataset(con)
    write_to_terminal("Completed the Steam game details dataset formatting !!!")

    con.close()
    write_to_terminal("Disconnected from the database to save memory !!!")


def formattedToTrusted():
    formatted_con = connectToDatabase('./Project/Formatted Zone/formatted_zone.duckdb')
    trusted_conn = connectToDatabase('./Project/Trusted Zone/trusted_zone.duckdb')
    write_to_terminal('Connected to the trusted database and formatted database !!!')

    root.after(500)
    epic_tables_to_combine = ['epic_games_v1', 'epic_games_v2']
    FormattedToTrusted.UnifyTables(epic_tables_to_combine,"epic_games", formatted_con, trusted_conn)
    write_to_terminal("Completed the Epic Games dataset unification !!!")

    root.after(500)
    steam_spy_tables_to_combine = ['steam_spy_v1','steam_spy_v2']
    FormattedToTrusted.UnifyTables(steam_spy_tables_to_combine,"steam_spy", formatted_con, trusted_conn)
    write_to_terminal("Completed the SteamSpy dataset unification !!!")

    root.after(500)
    steam_players_to_combine = ['steam_players_v1','steam_players_v2']
    FormattedToTrusted.UnifyTables(steam_players_to_combine,"steam_players", formatted_con, trusted_conn)
    write_to_terminal("Completed the Steam players dataset unification !!!")

    root.after(500)
    steam_game_info_to_combine = ['steam_game_info_v1','steam_game_info_v2']    
    FormattedToTrusted.UnifyTables(steam_game_info_to_combine,"steam_game_info", formatted_con, trusted_conn)
    write_to_terminal("Completed the Steam game info dataset unification !!!")

    trusted_conn.execute("DROP VIEW combined_data_df;")
    formatted_con.close()
    trusted_conn.close()
    write_to_terminal('Disconnected from the trusted database and formatted database to save memory !!!')


def trustedQuality():
    con = connectToDatabase('./Project/Trusted Zone/trusted_zone.duckdb')
    write_to_terminal('Connected to the trusted database for quality assessment!!!')

    root.after(500)
    TrustedQuality.epicGamesQuality(con)
    write_to_terminal("Completed the Epic Games dataset quality assessment !!!")

    
    root.after(500)
    TrustedQuality.steamCurrentPlayersQuality(con)
    write_to_terminal("Completed the Steam players dataset quality assessment !!!")

    root.after(500)
    TrustedQuality.steamSpyQuality(con)
    write_to_terminal("Completed the SteamSpy dataset quality assessment !!!")

    root.after(500)
    TrustedQuality.steamGameDetailsQuality(con)
    write_to_terminal("Completed the Steam game info dataset quality assessment !!!")
    
    con.close()
    write_to_terminal('Disconnected from the trusted database to save memory!!!')



def trustedToExploitation():
    root.after(1000)
    trusted_con = connectToDatabase('./Project/Trusted Zone/trusted_zone.duckdb')
    exploitation_conn = connectToDatabase('./Project/Exploitation Zone/exploitation_zone.duckdb')
    
    write_to_terminal('Connected to the trusted database and exploitation database!!!')

    root.after(500)
    TrustedToExploitation.mergeTables(trusted_con, exploitation_conn)
    write_to_terminal("Completed the merging of the datasets !!!")

    exploitation_conn.close()
    trusted_con.close()
    write_to_terminal('Disconnected from the trusted database and exploitation database to save memory !!!')

def exploitationQuality():
    con = connectToDatabase('./Project/Exploitation Zone/exploitation_zone.duckdb')
    write_to_terminal('Connected to the exploitation database for quality assessment!!!')

    root.after(500)
    ExploitationQualityKPI.mergeTableQuality(con)
    write_to_terminal("Completed the dataset quality assessment !!!")

    con.close()
    write_to_terminal('Disconnected from the exploitation database to save memory!!!')

def exploitationKPI():
    con = connectToDatabase('./Project/Exploitation Zone/exploitation_zone.duckdb')
    write_to_terminal('Connected to the exploitation database for KPI creation!!!')

    root.after(500)
    ExploitationQualityKPI.createKPIsTable(con)
    write_to_terminal("Completed the KPI creation table !!!")

    con.close()
    write_to_terminal('Disconnected from the exploitation database to save memory!!!')
def button1_action():
    write_to_terminal("Starting the Data Organization...")
    write_to_terminal("Loading data...")

    temporalToPersistent()
    ###########################################################
    persistentToFormatted()
    ################################################################
    formattedToTrusted()
    #####################################################################################################
    trustedQuality()
    ###########################################################################################
    trustedToExploitation()
    ###########################################################################################
    exploitationQuality()
    ###########################################################################################
    exploitationKPI()




    write_to_terminal("Process completed successfully !!!")
    start = True
    button1.config(state=tk.DISABLED)

    # time.sleep(5)  # Simulate a long-running process
    # button1.config(state=tk.NORMAL)


button1 = tk.Button(root, text="Start Data Organization", command=button1_action, width=20, height=2)
button1.place(x=10, y=10) 

# Create a Text widget to simulate the terminal output within the main window
terminal_text = tk.Text(root, bg="black", fg="white", font=("Courier", 10), wrap="word")
terminal_text.place(x=10, y=60, width=780, height=400)  

# Make terminal read-only initially
terminal_text.config(state=tk.DISABLED)

# Start the main loop
root.mainloop()



