import tkinter as tk
import tkinter.messagebox as messagebox
import duckdb
from Python_files import TemporalToPersistent
from Python_files import PersistentToFormatted
from Python_files import FormattedToTrusted
from Python_files import TrustedQuality
from Python_files import TrustedToExploitation
from Python_files import ExploitationQualityKPI
from Python_files.monitor_script import start_monitoring_thread, stop_monitoring_thread
from Python_files import CheckCodeQuality

def connectToDatabase(db_path):
    try:
        con = duckdb.connect(database=db_path, read_only=False)
    except Exception as e:
        write_to_terminal('Error connecting to the database with path'+ db_path +'and error message:'+ str(e))
        return None
    return con

def write_to_terminal(message):
    terminal_text.config(state=tk.NORMAL)  # Enable text widget to insert new text
    terminal_text.insert(tk.END, message + "\n")
    terminal_text.see(tk.END)  # Automatically scroll to the bottom
    terminal_text.config(state=tk.DISABLED)  # Disable editing again
    root.update()

def write_to_terminal2(message):
    terminal_text2.config(state=tk.NORMAL)
    terminal_text2.insert(tk.END, message + "\n")
    terminal_text2.see(tk.END)
    terminal_text2.config(state=tk.DISABLED)
    root.update()

def temporalToPersistent():
    root.after(500)
    TemporalToPersistent.create_folders()
    write_to_terminal("All the zone folders created !!!")

    root.after(500)
    succefull = TemporalToPersistent.move_files()
    if succefull:
        write_to_terminal("Completed the file organization !!!")
        return True
    else:
        write_to_terminal("Error in file organization !!!")
        return False


def persistentToFormatted():
    root.after(500)
    con = connectToDatabase('./Formatted Zone/formatted_zone.duckdb')
    if con is None:
        write_to_terminal("Error connecting to the database !!!")
        return False
    write_to_terminal("Connected to the database !!!")

    root.after(500)
    epicGamesFormatted = PersistentToFormatted.epicGamesDataset(con)
    if epicGamesFormatted:
        write_to_terminal("Completed the Epic Games dataset formatting !!!")
    else: 
        write_to_terminal("Error in Epic Games dataset formatting !!!")

    root.after(500)
    steamSpyFormatted = PersistentToFormatted.steamSpyDataset(con)
    if steamSpyFormatted:
        write_to_terminal("Completed the SteamSpy dataset formatting !!!")
    else:
        write_to_terminal("Error in SteamSpy dataset formatting !!!")

    root.after(500)
    steamCurrentPlayerFormatted = PersistentToFormatted.steamCurrentPlayerDataset(con)
    if steamCurrentPlayerFormatted:
        write_to_terminal("Completed the Steam current player dataset formatting !!!")
    else:
        write_to_terminal("Error in Steam current player dataset formatting !!!")

    root.after(500)
    steamDetailsFormatted = PersistentToFormatted.steamGameDetailsDataset(con)
    if steamDetailsFormatted: 
        write_to_terminal("Completed the Steam game details dataset formatting !!!")
    else:
        write_to_terminal("Error in Steam game details dataset formatting !!!")

    con.close()
    write_to_terminal("Disconnected from the database to save memory !!!")

    if epicGamesFormatted and steamSpyFormatted and steamCurrentPlayerFormatted and steamDetailsFormatted:
        return True
    else:
        return False


def formattedToTrusted():
    formatted_con = connectToDatabase('./Formatted Zone/formatted_zone.duckdb')
    if formatted_con is None:
        write_to_terminal("Error connecting to the formatted database !!!")
        return False
    write_to_terminal("Connected to the formatted database !!!")
    trusted_conn = connectToDatabase('./Trusted Zone/trusted_zone.duckdb')
    if trusted_conn is None:
        write_to_terminal("Error connecting to the trusted database !!!")
        return False
    write_to_terminal("Connected to the trusted database !!!")


    write_to_terminal("Starting the cleaning process...")

    root.after(500)
    epic_tables_to_combine = ['epic_games_v1', 'epic_games_v2']
    epicUnifyTable = FormattedToTrusted.UnifyTables(epic_tables_to_combine,"epic_games", formatted_con, trusted_conn)
    if epicUnifyTable:
        write_to_terminal("Completed the Epic Games dataset unification !!!")
    else:
        write_to_terminal("Error in Epic Games dataset unification !!!")

    root.after(500)
    steam_spy_tables_to_combine = ['steam_spy_v1','steam_spy_v2']
    steamSpyUnifyTable = FormattedToTrusted.UnifyTables(steam_spy_tables_to_combine,"steam_spy", formatted_con, trusted_conn)
    if steamSpyUnifyTable:
        write_to_terminal("Completed the SteamSpy dataset unification !!!")
    else:
        write_to_terminal("Error in SteamSpy dataset unification !!!")

    root.after(500)
    steam_players_to_combine = ['steam_players_v1','steam_players_v2']
    steamPlayersUnifyTable = FormattedToTrusted.UnifyTables(steam_players_to_combine,"steam_players", formatted_con, trusted_conn)
    if steamPlayersUnifyTable:
        ("Completed the Steam players dataset unification !!!")
    else:
        write_to_terminal("Error in Steam players dataset unification !!!")

    root.after(500)
    steam_game_info_to_combine = ['steam_game_info_v1','steam_game_info_v2']    
    steamGameUnifyTable = FormattedToTrusted.UnifyTables(steam_game_info_to_combine,"steam_game_info", formatted_con, trusted_conn)
    if steamGameUnifyTable:
        write_to_terminal("Completed the Steam game info dataset unification !!!")
    else:
        write_to_terminal("Error in Steam game info dataset unification !!!")

    trusted_conn.execute("DROP VIEW combined_data_df;")
    formatted_con.close()
    trusted_conn.close()
    write_to_terminal('Disconnected from the trusted database and formatted database to save memory !!!')

    if epicUnifyTable and steamSpyUnifyTable and steamPlayersUnifyTable and steamGameUnifyTable:
        return True
    else:
        return False


def trustedQuality():
    con = connectToDatabase('./Trusted Zone/trusted_zone.duckdb')
    if con is None:
        write_to_terminal("Error connecting to the trusted database !!!")
        return False
    write_to_terminal('Connected to the trusted database for quality assessment!!!')

    root.after(500)
    epicGamesQuality = TrustedQuality.epicGamesQuality(con)
    if epicGamesQuality:
        write_to_terminal("Completed the Epic Games dataset quality assessment !!!")
    else:
        write_to_terminal("Error in Epic Games dataset quality assessment !!!")

    
    root.after(500)
    steamCurrentPlayersQuality = TrustedQuality.steamCurrentPlayersQuality(con)
    if steamCurrentPlayersQuality:
        write_to_terminal("Completed the Steam players dataset quality assessment !!!")
    else:
        write_to_terminal("Error in Steam players dataset quality assessment !!!")

    
    root.after(500)
    steamSpyQuality = TrustedQuality.steamSpyQuality(con)
    if steamSpyQuality:
        write_to_terminal("Completed the SteamSpy dataset quality assessment !!!")
    else:
        write_to_terminal("Error in SteamSpy dataset quality assessment !!!")


    root.after(500)
    steamDetailQuality = TrustedQuality.steamGameDetailsQuality(con)
    if steamDetailQuality:
        write_to_terminal("Completed the Steam game info dataset quality assessment !!!")
    else:
        write_to_terminal("Error in Steam game info dataset quality assessment !!!")
    
    con.close()
    write_to_terminal('Disconnected from the trusted database to save memory!!!')
    if epicGamesQuality and steamCurrentPlayersQuality and steamSpyQuality and steamDetailQuality:
        return True
    return False

def trustedToExploitation():
    root.after(500)
    trusted_con = connectToDatabase('./Trusted Zone/trusted_zone.duckdb')
    if trusted_con is None:
        write_to_terminal("Error connecting to the trusted database !!!")
        return False
    write_to_terminal('Connected to the trusted database !!!')

    exploitation_conn = connectToDatabase('./Exploitation Zone/exploitation_zone.duckdb')
    if exploitation_conn is None:
        write_to_terminal("Error connecting to the exploitation database !!!")
        return False
    write_to_terminal('Connected to the exploitation database !!!')
    

    root.after(500)
    tablesMerged = TrustedToExploitation.mergeTables(trusted_con, exploitation_conn)
    if tablesMerged:
        write_to_terminal("Completed the merging of the datasets !!!")
    else: 
        write_to_terminal('Error in merging the datasets !!!')

    exploitation_conn.close()
    trusted_con.close()
    write_to_terminal('Disconnected from the trusted database and exploitation database to save memory !!!')
    return tablesMerged

def exploitationQuality():
    con = connectToDatabase('./Exploitation Zone/exploitation_zone.duckdb')
    if con is None:
        write_to_terminal("Error connecting to the exploitation database !!!")
        return False
    write_to_terminal('Connected to the exploitation database for quality assessment!!!')

    root.after(500)
    mergeTableQuality = ExploitationQualityKPI.mergeTableQuality(con)
    if mergeTableQuality:
        write_to_terminal("Completed the dataset quality assessment !!!")
    else:
        write_to_terminal("Error in dataset quality assessment !!!")

    con.close()
    write_to_terminal('Disconnected from the exploitation database to save memory!!!')
    return mergeTableQuality

def exploitationKPI():
    con = connectToDatabase('./Exploitation Zone/exploitation_zone.duckdb')
    if con is None:
        write_to_terminal("Error connecting to the exploitation database !!!")
        return False
    write_to_terminal('Connected to the exploitation database for KPI creation!!!')

    root.after(500)
    createdKPIsTable = ExploitationQualityKPI.createKPIsTable(con)
    if createdKPIsTable:
        write_to_terminal("Completed the KPI creation table !!!")
    else:
        write_to_terminal("Error in KPI creation table !!!")

    con.close()
    write_to_terminal('Disconnected from the exploitation database to save memory!!!')
    return createdKPIsTable

def button1_action():
    global start
    if not start:
        button1.config(state=tk.DISABLED)
        write_to_terminal("Starting the Data Organization...")
        write_to_terminal("Loading data...")

        global monitor_active, monitor_thread
        if not monitor_active:
            write_to_terminal2("Monitoring process started...")
            monitor_thread = start_monitoring_thread(write_to_terminal2)  
            monitor_active = True
        else:
            messagebox.showinfo("Proceso en Ejecución", "El proceso de monitoreo ya está en ejecución.")
        


        sucessful = temporalToPersistent()
        if not sucessful: 
            write_to_terminal("Failed to organize the data. Please check the terminal output for more details.")
            return
        else:
            sucessful = persistentToFormatted()
        
        if not sucessful:
            write_to_terminal("Failed to format the data. Please check the terminal output for more details.")
            return
        else:
            sucessful = formattedToTrusted()
        
        if not sucessful:
            write_to_terminal("Failed to transform the data. Please check the terminal output for more details.")
            return
        else:
            sucessful = trustedQuality()
        
        if not sucessful:
            write_to_terminal("Failed to assess the quality of the data. Please check the terminal output for more details.")
            return
        else:
            sucessful = trustedToExploitation()
        
        if not sucessful:
            write_to_terminal("Failed to merge the data. Please check the terminal output for more details.")
            return
        else:
            sucessful = exploitationQuality()
            
        if not sucessful:
            write_to_terminal("Failed to assess the quality of the merged data. Please check the terminal output for more details.")
            return
        else:
            sucessful = exploitationKPI()

        if not sucessful:
            write_to_terminal("Failed to create the KPIs. Please check the terminal output for more details.")
            return
        else:
            write_to_terminal("All processes completed successfully !!!")
            root.after(500)
            write_to_terminal("Data organization process completed successfully !!!")
            start = True
            button1.config(state=tk.NORMAL)
        stop_monitoring_thread()

        
    else:
        messagebox.showinfo("Process Completed", "The process has already been completed.")



def doChekCodeQuality(secondary_terminal):
    
    secondary_terminal.config(state=tk.NORMAL)
    secondary_terminal.insert(tk.END, "Checking Code Quality...\n")
        
    code_quality = CheckCodeQuality.run_flake8() 
    
    secondary_terminal.insert(tk.END, code_quality)
    secondary_terminal.insert(tk.END, "\n")
    secondary_terminal.insert(tk.END, "Finished checking code quality.")
    secondary_terminal.see(tk.END)
    secondary_terminal.config(state=tk.DISABLED)

# Open new window with code quality check
def button2_action():
    secondary_window = tk.Toplevel(root)
    secondary_window.title("Code Quality Check")
    secondary_window.geometry("600x500")

    label = tk.Label(secondary_window, font=("Arial", 16), text="Checking code quality...")
    label.place(relx=0.5, rely=0.05, anchor="center") # Centered at the top of the window


    secondary_terminal = tk.Text(secondary_window, bg="black", fg="white", font=("Courier", 10), wrap="word")
    secondary_terminal.place(relx=0.05, rely=0.15, relwidth=0.9, relheight=0.8)  # Fills most of the window

    secondary_window.after(500, lambda: doChekCodeQuality(secondary_terminal))
    


# Create the main window
root = tk.Tk()
root.title("Data Management Backbone")
root.geometry("800x500")  # Set the size of the window

start = False
monitor_thread = None 
monitor_active = False  


button1 = tk.Button(root, text="Start Data Organization", command=button1_action, width=20, height=2)
button1.place(relx=0.01, rely=0.02, relwidth=0.4, relheight=0.1)

button2 = tk.Button(root, text="Check Code Quality", command=button2_action, width=20, height=2)
button2.place(relx=0.59, rely=0.02, relwidth=0.4, relheight=0.1)

# Create a Text widget to simulate the terminal output within the main window
terminal_text = tk.Text(root, bg="black", fg="white", font=("Courier", 10), wrap="word")
terminal_text.place(relx=0.01, rely=0.15, relwidth=0.48, relheight=0.8)  

terminal_text2 = tk.Text(root, bg="black", fg="white", font=("Courier", 10), wrap="word")
terminal_text2.place(relx=0.51, rely=0.15, relwidth=0.48, relheight=0.8)  
write_to_terminal("Click the 'Start Data Organization' button to start the data organization process :)")

# Make terminal read-only initially
terminal_text.config(state=tk.DISABLED)

# Start the main loop
root.mainloop()





