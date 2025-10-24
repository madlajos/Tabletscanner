import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SqlDatabaseComponent } from './sql-database.component';

describe('SqlDatabaseComponent', () => {
  let component: SqlDatabaseComponent;
  let fixture: ComponentFixture<SqlDatabaseComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SqlDatabaseComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SqlDatabaseComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
