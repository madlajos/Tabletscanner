import { Component, OnInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { interval, Subscription, of } from 'rxjs';
import { switchMap, catchError, timeout } from 'rxjs/operators';
import { ErrorNotificationService } from '../../services/error-notification.service';


interface SQLServerSettings {
  server: string;
  db_name: string;
  username: string;
  password: string;
}

@Component({
  selector: 'app-sql-database',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './sql-database.component.html',
  styleUrls: ['./sql-database.component.css']
})

export class SQLDatabaseComponent implements OnInit, OnDestroy {
  private readonly BASE_URL = 'http://localhost:5000/api';

  // Local model for SQL Server settings.
  server: string = "";
  db_name: string = "";
  username: string = "";
  password: string = "";

  // Connection state.
  connected: boolean = false;
  connectionStatus: string = "Not Connected";

  private connectionPolling: Subscription | undefined;
  // Unified polling subscription for SQL settings:
  private sqlSettingsPollingSub: Subscription | undefined;

  constructor(
    private http: HttpClient,
    private cdr: ChangeDetectorRef,
    private errorNotificationService: ErrorNotificationService
  ) {}

  ngOnInit(): void {
    // Load SQL settings from backend.
     this.http.get<{ sql_server: any }>(`${this.BASE_URL}/get-other-settings?category=sql_server`)
       .subscribe({
         next: response => {
           if (response && response.sql_server) {
             this.server = response.sql_server.server;
             this.db_name = response.sql_server.db_name;
             this.username = response.sql_server.username;
             this.password = response.sql_server.password;
           }
         },
         error: err => {
           console.error("Failed to load SQL Server settings:", err);
         }
       });
    

    // Immediately try to connect to the database.
    this.connectDatabase();

    // Start polling the connection status every 5 seconds.
    this.startConnectionPolling();
  }

  ngOnDestroy(): void {
    if (this.connectionPolling) {
      this.connectionPolling.unsubscribe();
    }
    if (this.sqlSettingsPollingSub) {
      this.sqlSettingsPollingSub.unsubscribe();
    }
  }

  // Update a specific SQL setting.
  updateSetting(settingName: string, value: any): void {
    this.http.post(`${this.BASE_URL}/update-other-settings`, {
      category: "sql_server",
      setting_name: settingName,
      setting_value: value
    }).subscribe({
      next: (response: any) => {
        console.log(`Updated ${settingName}:`, response);
      },
      error: err => {
        console.error(`Failed to update ${settingName}:`, err);
      }
    });
  }

  // Called when the user changes a textbox.
  onSettingChange(settingName: string, event: any): void {
    const newValue = event.target.value;
    this.updateSetting(settingName, newValue);
  }

  // Connect to the SQL database.
  connectDatabase(): void {
    this.http.get<{ message?: string, error?: string }>(`${this.BASE_URL}/connect_sql_database`)
      .subscribe({
        next: response => {
          if (response.message) {
            this.connectionStatus = response.message;
            this.connected = true;
            this.errorNotificationService.removeError("E1401");
          }
          this.cdr.detectChanges();
        },
        error: err => {
          console.error("Failed to connect to SQL database:", err);
          this.connectionStatus = err.error?.error || "Connection failed";
          this.connected = false;
          this.errorNotificationService.addError({
            code: "E1401",
            message: this.errorNotificationService.getMessage("E1401")
          });
          this.cdr.detectChanges();
        }
      });
  }

  // Disconnect from the SQL database.
  disconnectDatabase(): void {
    this.http.post(`${this.BASE_URL}/disconnect_sql_database`, {}).subscribe({
      next: response => {
        console.log("Disconnected from SQL database:", response);
        this.connected = false;
        this.connectionStatus = "Disconnected";
        this.errorNotificationService.addError({
          code: "E1401",
          message: this.errorNotificationService.getMessage("E1401")
        });
        this.cdr.detectChanges();
      },
      error: err => {
        console.error("Failed to disconnect SQL database:", err);
      }
    });
  }

  // Toggle connection.
  toggleConnection(): void {
    if (this.connected) {
      this.disconnectDatabase();
    } else {
      this.connectDatabase();
    }
  }

  startConnectionPolling(): void {
    if (!this.connectionPolling) {
      this.connectionPolling = interval(5000).pipe(
        switchMap(() =>
          // Append a timestamp to avoid caching.
          this.http.get<{ message?: string, error?: string }>(
            `${this.BASE_URL}/check-db-connection?ts=${new Date().getTime()}`
          ).pipe(
            timeout(3000),
            catchError(err => {
              console.error("SQL Database polling encountered an error:", err);
              return of({ message: "", error: err.error?.error || "Connection failed" });
            })
          )
        )
      ).subscribe({
        next: response => {
          if (response.message && response.message.trim() !== "") {
            if (!this.connected) {
              console.info("SQL Database reconnected.");
              this.errorNotificationService.removeError("E1401");
            }
            this.connectionStatus = response.message;
            this.connected = true;
          } else if (response.error) {
            console.error("SQL Database connection polling error response:", response.error);
            this.connectionStatus = response.error;
            this.connected = false;
            this.errorNotificationService.addError({
              code: "E1401",
              message: this.errorNotificationService.getMessage("E1401")
            });
          }
          this.cdr.detectChanges();
        },
        error: err => {
          console.error("SQL Database connection polling error (final):", err);
          const fallbackError = "Connection failed";
          const errMsg = (err.error && err.error.error) ? err.error.error : (err.message || fallbackError);
          this.connectionStatus = errMsg;
          this.connected = false;
          this.errorNotificationService.addError({
            code: "E1401",
            message: this.errorNotificationService.getMessage("E1401")
          });
          this.cdr.detectChanges();
        }
      });
    }
  }
}
