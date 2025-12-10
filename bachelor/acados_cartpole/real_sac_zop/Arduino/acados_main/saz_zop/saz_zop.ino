/*
-nov 24: heavily simplified position control. Position only comes from AB counts potenzial drift now
*/


#include <Wire.h>
#include <AS5600.h>

//AS5600
AS5600 as5600(&Wire);

//Motor
const int PWM_PIN = 5;
const int DIR_PIN = 6; // LOW is to the right currently

// GHH60
constexpr uint8_t PIN_A = 2;
constexpr uint8_t PIN_B = 3;

constexpr long cpr_ghh60 = 1024; 
constexpr float TAU_V = 0.005f;  // 10 ms
constexpr uint32_t position_limit = 11000;

volatile bool tripped = false; // flag that indicates whether we exceeded the position limit
volatile long x = 0; // cartpostion in coutns
volatile long v = 0; // cartposition in counts/s
volatile long encoder_pos = 0; 

//AS5600
constexpr long cpr_as5600 = 4096;
constexpr float two_pi = 2.0f * PI;
constexpr float four_pi = 4.0f * PI;
constexpr float step_rad = (2.0f * PI) / cpr_as5600;
constexpr float TAU_S = 0.005; 

static float last_theta = 0.0f;
volatile uint16_t zero_raw_top = 0;
volatile uint16_t last_raw = 0;
volatile float angle_unwrapped = 0.0f;
volatile float thetadot = 0.0f;

//communication and hyperparameters
constexpr long BAUDRATE = 115200;
constexpr unsigned long COMMUNICATION_TIME_MS = 15;
constexpr uint32_t STATE_UPDATE_US = 5000;
constexpr byte NUM_CHARS = 32;
constexpr long X_CENTER_COUNTS = 0;      // ceter position in counts
constexpr long X_CENTER_TOL    = 100;    // tolerance around center position



volatile uint8_t mode = 0; 
volatile bool new_data = false; // if u has been send to motor -> false
volatile char receivediff_cnthars[NUM_CHARS];
volatile float u = 0.0f;

//Motor tuning
constexpr int U_MAX        = 255;  // max allowed PWM
constexpr int PWM_DEADBAND = 3;    // pwm <= will be ignored
constexpr int PWM_MIN_EFF  = 4;   // min pwm to overcome friction

// merkt sich die letzte Richtung, um Deadtime nur bei Richtungswechsel zu setzen
static int last_sign = 0;



// ISR-------------------------------------------------------------------------------
void isrA() {
  /*
  -currently triggered on RISING
  -count A flanks from GHH60
  */

  if (digitalRead(PIN_B)) {
    encoder_pos--;
  }
  else {
    encoder_pos++;
  }
}



//COMMUNICATION-------------------------------------------------------------------------
void receive_parse_data() {
  /*
    Parses and receives in the form: "<signedint, boolean/int>"
    Example: "<120,1>"
  */

  static bool in_progress = false;
  static byte idx = 0;
  
  while (Serial.available() > 0) {
    char rc = Serial.read();

    if (in_progress == true) {
      if (rc == '>') {
        // frame finished -> reset everything and terminate
        receivediff_cnthars[idx] = '\0';
        idx = 0;
        in_progress = false;

        // set the u command and new_data = true
        char *comma = strchr((char *)receivediff_cnthars, ',');
        if (comma != NULL) {
          *comma = '\0';                  // slice the string at the comma
          const char *u_str    = (const char *)receivediff_cnthars;
          const char *mode_str = (const char *)(comma + 1);

          // mode parsen
          mode = atoi(mode_str);          // 0 / 1 / 2

          // only if normal SAC mode, set u otherwise ignore
          if (mode == 0) {
            // normal SAC mode
            u    = atoi(u_str);             // in PWM units but signed
          }
      
          new_data = true;
        }
      }
      else {
        if (rc == '\r' || rc == '\n') continue; // handle errors
        if (idx + 1 < NUM_CHARS) {
          // we currently read the payload
          receivediff_cnthars[idx] = rc;
          idx++;
        } 
      }
    }
    else {
      if (rc == '<') {
        in_progress = true;
        idx = 0;
      }
    }
  }
}


void send_state(long x, float theta, long v,  float thetadot, bool tripped) {
  /*
  -sends a state as a string in the form 
  */

  float t = millis() * 0.001f; // in seconds
  
  Serial.write('<'); // startmarker
  Serial.print(x);         
  Serial.write(',');
  Serial.print(theta);
  Serial.write(',');
  Serial.print(v);
  Serial.write(',');
  Serial.print(thetadot, 3);
  Serial.write(',');
  Serial.print(tripped);
  Serial.write('>'); // endmarker
  Serial.write('\n'); // only for debugging  
}



//HELPERS--------------------------------------------------------------------------------
inline void update_theta_omega(const float T) {
  /*
  -updates the value of theta
  -and calculates omega 
  */

  // read i2c
  const uint16_t raw = readRaw();
  const int16_t  dr  = delta_from_raw(raw, last_raw);
  last_raw           = raw;

  // make angle continuous by always sub/add; incremental step
  const float dtheta = ((float)dr * step_rad);
  angle_unwrapped   += dtheta;

  // raw theta-dot taken (no filtering)
  float thetadot_raw = dtheta / T; // rad/s

  // EMA smoothing for theta-dot (seed on first call)
  static float thetadot_hat = 0.0f;
  static bool thetadot_hat_initialized = false;
  const float alpha = T / (TAU_S + T);
  if (!thetadot_hat_initialized) {
    thetadot_hat = thetadot_raw;
    thetadot_hat_initialized = true;
  } else {
    thetadot_hat = (1.0f - alpha) * thetadot_hat + alpha * thetadot_raw;
  }
  thetadot = thetadot_hat;
}


inline void update_x_v(const float T) {
  /*
  -gets the absolute position of the cart and assigns it to x
  -calculates v with it
  */
  
  const long cnt = encoder_pos;

  // assign cnt to x
  x = cnt;

  if (abs(x) > position_limit) {
    tripped = true;
  }
  // calculate the raw velocity
  static long last_cnt = cnt;      
  const long diff_cnt = cnt - last_cnt;
  last_cnt = cnt;

  
  const float v_raw = (float)diff_cnt / T;   // counts/s

  // apply EMA filter to it
  static float v_hat = 0.0f;
  const float alpha = T / (TAU_V + T);
  v_hat = v_hat + alpha * (v_raw - v_hat);

  // assign it to v
  v = (long)lroundf(v_hat);
}

uint16_t readRaw() {
  /* 
  -reads the angle (0-4096) from AS5600
  */

  return as5600.rawAngle() & 0x0FFF;
}


int16_t delta_from_raw(uint16_t nowRaw, uint16_t last_raw_local) {
  /*
  -corrects the the jump between 0/4096
  -makes the function smooth and so differentiable
  */

  int16_t diff = (int16_t)nowRaw - (int16_t)last_raw_local;
  if (diff > 2048) {
    diff = diff - 4096;
  }
  if (diff < -2048) {
    diff = diff + 4096;
  }

  return diff;
}


float wrap_to_twopi(float a) {
  /*
  -works but has to be understood
  -wraps the absolute postion of the magnetic ecoder to the desired intervall
  */

  a = fmodf(a, four_pi);        // (-4π, 4π)
  if (a <  -two_pi) {
    a += four_pi;
  }
  if (a > two_pi) {
    a -= four_pi;
  }
  return a;                     // [-2π, +2π]
}


float wrap_to_pi(float a) {
  // wrap nach [-pi, pi]
  a = fmodf(a + PI, two_pi);   // in [0, 2pi)
  if (a < 0) a += two_pi;      // numerische Sicherheit
  return a - PI;               // in [-pi, pi]
}


float raw_to_rad(uint16_t raw) {
  /*
  -only called once
  */

  uint16_t diff = (raw + cpr_as5600- zero_raw_top) & 0x0FFF;
  // convert raw encoder diff to radians (no sign inversion)
  return diff * step_rad;
}


void apply_u() {
  // dobble check tripped condition
  if (tripped) {
  analogWrite(PWM_PIN, 0);
  return;
  }
  
  int u_cmd = (int)lroundf(u);
  u_cmd = constrain(u_cmd, -U_MAX, U_MAX);

  // determine sign explicitly
  int sign;
  if (u_cmd > 0) {
    sign = 1;
  } else if (u_cmd < 0) {
    sign = -1;
  } else {
    sign = 0;
  }

  // magnitude (absolute value)
  int mag = abs(u_cmd);

  // handle deadband: very small commands are ignored
  if (mag < PWM_DEADBAND) {
    analogWrite(PWM_PIN, 0);
    return;
  }
  if (mag < PWM_MIN_EFF) {
    mag = PWM_MIN_EFF;
  }
  // set the direction accordingly
  // a short deadtime when the sign (direction) changes.
  if (sign != last_sign) {
    // stop PWM before changing direction
    analogWrite(PWM_PIN, 0);
    delayMicroseconds(50);
  }

  // Consistent direction mapping: positive -> LOW, negative -> HIGH
  if (sign > 0) {
    digitalWrite(DIR_PIN, LOW);
  } else if (sign < 0) {
    digitalWrite(DIR_PIN, HIGH);
  }

  // small settle delay after (possible) direction change
  delayMicroseconds(50);
  last_sign = sign;
  analogWrite(PWM_PIN, mag);
}


void motorstop() {
  /*
  -terminates the Motor
  -used currently only for safety watchdog
  */

  analogWrite(PWM_PIN, 0);
}


void reset_control() {
  /*
  - simple rule to bring the cart back to center position (x=0)
  - erro > 0 -> need to go right
  - error < 0 -> need to go left
  */

  long x_local = x;
  long error   = X_CENTER_COUNTS - x_local;

  if (labs(error) < X_CENTER_TOL && 
      fabs(thetadot) < 0.1f && 
      fabsf(abs(wrap_to_pi(angle_unwrapped))) >= 3.0f) {
    // we are ready to go. Cart in the middle end Pendulum not moving
    u = 0.0f;     
    tripped = false;  
    return;
  }

  else if (labs(error) < X_CENTER_TOL) {
    // we are in the middle but pendulum is moving
    motorstop();
    return;
  }

  else {
    // cart not in the middle
    if (error >= 0) {
      // need to go right
      analogWrite(PWM_PIN, 25); // stop motor before changing direction
      digitalWrite(DIR_PIN, LOW);
    } else {
      // need to go left
      analogWrite(PWM_PIN, 25); // stop motor before changing direction
      digitalWrite(DIR_PIN, HIGH);
    }
    
  }
}



void setup() {
  // begin serial communication
  Serial.begin(BAUDRATE);
  while (!Serial) {}
  
  // Motor PINS
  pinMode(PWM_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);

  // i2c init
  Wire.begin();
  Wire.setClock(100000);

  
  if (!as5600.begin()) {  //test if the sensor go recognized 
    Serial.println("AS5600 not found");
    while (1) { delay(1000); }
  }
  
  // wait for pendel osilations to drop
  delay(5000);

  // Magnetic Encoder init
  uint16_t zero_raw_bottom = readRaw();   

  zero_raw_top = (zero_raw_bottom + (cpr_as5600 / 2)) & 0x0FFF;

  last_raw = readRaw();

  angle_unwrapped = raw_to_rad(last_raw);
  
  //Motor init
  digitalWrite(DIR_PIN, LOW);
  analogWrite(PWM_PIN, 0);

  //GHH60 Sensor PINS
  pinMode(PIN_A, INPUT);
  pinMode(PIN_B, INPUT);

  // ISR
  attachInterrupt(digitalPinToInterrupt(PIN_A), isrA, RISING);

  // startup test move
  digitalWrite(DIR_PIN, LOW); // first direction
  analogWrite(PWM_PIN, 25);
  delay(50);
  analogWrite(PWM_PIN, 0);
  delay(50);
  digitalWrite(DIR_PIN, HIGH); // reverse direction
  analogWrite(PWM_PIN, 25);
  delay(50);
  analogWrite(PWM_PIN, 0);
  delay(50);

  // pause before start
  delay(4000);

}


void loop() {
  // receive new data
  receive_parse_data();

  // mode handling. 0 = normal SAC, 1 = stop, 2 = reset environment
  if (mode == 2) {
    reset_control();
  }

  else if (mode == 1) {
    motorstop();
    new_data = false;
  }

  else {
    // normal SAC mode
    // do nothing here, u has already been applied above
    if (tripped) {
      motorstop();
      new_data = false;
    } else {
      if (new_data) {
        apply_u();
        new_data = false;
      }
    }
  }

  // update the statevector 
  static uint32_t time_last_angle = 0;
  uint32_t nowu = micros();

  if ((uint32_t)(nowu - time_last_angle) >= STATE_UPDATE_US) {
    // make it robust again jitter
    uint32_t periods = 0;
    do { time_last_angle += STATE_UPDATE_US; periods++; }
    while ((uint32_t)(nowu - time_last_angle) >= STATE_UPDATE_US);

    // real delta t. This version is more robust against jitters
    const float real_dt = periods * (STATE_UPDATE_US * 1e-6f);

    // update the states
    update_theta_omega(real_dt);
    update_x_v(real_dt);
  }


  // main 10ms controll loop
  static unsigned long next_ms = 0;
  unsigned long now = millis();
  if ((long)(now - next_ms) >= 0) {
    do { next_ms += COMMUNICATION_TIME_MS; } while ((long)(now - next_ms) >= 0);

  
    float theta = angle_unwrapped;
    // send the state here: x, theta, v, thetadot (raw derivative)
    // flip sign of theta and thetadot to match external convention
    send_state(x, -wrap_to_pi(theta), v, -thetadot, tripped);

  }
}