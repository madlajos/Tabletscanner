import { provideHttpClient } from '@angular/common/http';
import { TestBed } from '@angular/core/testing';

import { AppError, ErrorNotificationService } from './error-notification.service';

describe('ErrorNotificationService', () => {
  let service: ErrorNotificationService;
  let errors: AppError[];

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient()],
    });
    service = TestBed.inject(ErrorNotificationService);
    errors = [];
    service.errors$.subscribe(value => errors = value);
  });

  it('shows homing failures as centered popups without marking them as critical alerts', () => {
    service.addError({ code: 'E1202', message: 'Homing failed' });

    expect(errors).toEqual([
      jasmine.objectContaining({
        code: 'E1202',
        popupStyle: 'center',
      }),
    ]);
    expect(errors[0].abortMeasurement).toBeUndefined();
  });

  it('keeps motion-platform disconnects in the default critical alert list', () => {
    service.addError({ code: 'E1201', message: 'Disconnected' });

    expect(errors).toEqual([
      jasmine.objectContaining({
        code: 'E1201',
      }),
    ]);
    expect(errors[0].popupStyle).toBeUndefined();
  });
});
