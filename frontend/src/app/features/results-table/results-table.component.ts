import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SharedService, MeasurementRecord } from '../../shared.service';
import { Subscription } from 'rxjs';

interface MeasurementResult {
  date: string;
  time: string;
  id: number | string;
  barcode: string;
  operator: string;
  clogged: number | string;
  partiallyClogged: number | string;
  clean: number | string;
  result: string;
}

@Component({
  standalone: true,
  selector: 'app-results-table',
  imports: [CommonModule],
  templateUrl: './results-table.component.html',
  styleUrls: ['./results-table.component.css']
})
export class ResultsTableComponent implements OnInit, OnDestroy {
  results: MeasurementResult[] = [];
  private subscription!: Subscription;

  constructor(private sharedService: SharedService) {}

  ngOnInit(): void {
    // Subscribe to measurement history updates from SharedService.
    this.subscription = this.sharedService.measurementHistory$.subscribe(
      (data: MeasurementRecord[]) => {
        this.results = data;
      }
    );
  }
  
  ngOnDestroy(): void {
    // Unsubscribe to prevent memory leaks.
    if (this.subscription) {
      this.subscription.unsubscribe();
    }
  }
}