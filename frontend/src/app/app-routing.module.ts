import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { ImageViewerComponent } from './features/image-viewer/image-viewer.component';
import { CameraControlComponent } from './features/camera-control/camera-control.component';

const routes: Routes = [
  { path: ''},  // Default route
  { path: 'camera-control', component: CameraControlComponent },  // Camera Control page
  { path: '**', redirectTo: '' }  // Redirect unknown routes to home
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {}
