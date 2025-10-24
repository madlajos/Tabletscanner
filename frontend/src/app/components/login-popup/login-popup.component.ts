import { Component, EventEmitter, Input, Output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-login-popup',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './login-popup.component.html',
  styleUrls: ['./login-popup.component.css']
})
export class LoginPopupComponent {
  @Input() isVisible: boolean = false;
  @Input() loginError: string = '';
  @Output() loginSuccess = new EventEmitter<{ username: string; password: string }>();
  @Output() closePopup = new EventEmitter<void>();

  username: string = '';
  password: string = '';

  private readonly correctUsername = 'admin';
  private readonly correctPassword = '1234';

  login(): void {
    if (this.username === this.correctUsername && this.password === this.correctPassword) {
      this.loginError = '';
      // Emit the login success event with the credentials if needed.
      this.loginSuccess.emit({ username: this.username, password: this.password });
      // Clear the input fields so that they're empty next time the popup opens.
      this.username = '';
      this.password = '';
      // Close the popup
      this.hide();
    } else {
      this.loginError = 'Hibás felhasználónév vagy jelszó!';
    }
  }
  
  hide(): void {
    this.closePopup.emit();
  }
  
}
