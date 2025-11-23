void isrA() {
  /*
  -currently triggered on RISING
  -count A flanks from GHH60
  */

  if (digitalRead(PIN_B)) {
    encoder_pos--;
    dir_from_ab = -1;
  }
  else {
    encoder_pos++;
    dir_from_ab = 1;
  }
}


void isrZ() {
  /*
  -triggered on RISING z edge from GHH60
  -used to reset position to avoid drift
  -with safety mechanism to avoid misscounting
  */

  unsigned long now = micros();
  if ((unsigned long)(now - lastZMicros) < (unsigned long)Z_LOCKOUT_US) {
    return;
  }
  lastZMicros = now;

  encoder_pos = 0;

  int8_t dir = dir_from_derivative;  // Primär: Richtung aus den Counts

  // Fallback, falls in der letzten Ableitungsperiode keine Bewegung war
  if (dir == 0) {
    dir = dir_from_ab;
  }

  // Zähler entsprechend der (jetzt bestimmten) Richtung anpassen
  if (dir >= 0) {
    z_counter++;
  } else {
    z_counter--;
  }

  //only called once for initialisation
  if (z_primed == false) {
    z_counter = 0;
    x = 0;
    z_primed = true;
  }

}



inline void update_x_v(const float T) {
  /*
  -gets the absolute position of the cart and assigns it to x
  -calculates v with it
  */
  
  const long cnt = absolute_counts();

  // assign cnt to x
  x = cnt;

  if (abs(x) > position_limit) {
    tripped = true;
  }
  // calculate the raw velocity
  static long last_cnt = cnt;      
  const long dc = cnt - last_cnt;
  last_cnt = cnt;

  // calculate direction for z isr
  if      (dc > 0) dir_from_derivative = +1;
  else if (dc < 0) dir_from_derivative = -1;
  
  const float v_raw = (float)dc / T;   // counts/s

  // apply EMA filter to it
  static float v_hat = 0.0f;
  const float alpha = T / (TAU_V + T);
  v_hat = v_hat + alpha * (v_raw - v_hat);

  // assign it to v
  v = (long)lroundf(v_hat);
}

long absolute_counts() {
  /*
  -calculates the exact position of the cart in relation to the zero point
  */

  noInterrupts();
  long z = z_counter;
  long p = encoder_pos;
  interrupts();

  // p in [0..CPR-1] falten (nur Darstellung)
  if (p < 0) p += cpr_ghh60;
  else if (p >= cpr_ghh60) p -= cpr_ghh60;

  long raw = z * cpr_ghh60 + p;

  static long last = raw;     // init beim ersten Aufruf
  long diff = raw - last;

  // Sprung-Korrektur: „zieh“ raw an last heran
  long half = cpr_ghh60 / 2;
  if (diff >  half) raw -= cpr_ghh60;
  else if (diff < -half) raw += cpr_ghh60;

  last = raw;

  return raw;
}

