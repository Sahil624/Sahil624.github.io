import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { HomeComponent } from './home.component';
import { LandingComponent } from './components/landing/landing.component';


@NgModule({
  declarations: [
      HomeComponent,
      LandingComponent
  ],
  imports: [
    CommonModule
  ],
  exports: [
      HomeComponent
  ]
})
export class HomeModule { }
