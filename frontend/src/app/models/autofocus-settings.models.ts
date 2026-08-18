import { LightChannel, UvBrightnessMode } from './light.models';

export interface AutofocusSettings {
  channel: LightChannel;
  brightness: UvBrightnessMode;
  filter_position: 1 | 2 | 3 | 4 | 5 | 6;
}

export interface AutofocusLightOption {
  channel: LightChannel;
  brightness: UvBrightnessMode;
  label: string;
}

export interface AutofocusFilterOption {
  position: 1 | 2 | 3 | 4 | 5 | 6;
  label: string;
  unavailable?: boolean;
}
