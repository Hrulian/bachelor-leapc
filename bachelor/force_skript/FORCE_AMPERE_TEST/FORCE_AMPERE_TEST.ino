/*
TODO:
-safetyISR muss noch fertig gemacht werden 
-kraft des mpc in ampere
*/


#include <DueTimer.h>
#include <Wire.h>
#include <AS5600.h>

//AS5600
AS5600 as5600(&Wire);

//Pins
//GHH60 Sensor
constexpr uint8_t PIN_A = 2;
constexpr uint8_t PIN_B = 3;
constexpr uint8_t PIN_Z = 4;

//Motor
const int PWM_PIN = 5;
const int DIR_PIN = 6; // LOW is to the right

#define ZERO_ON_Z_RISING   1
#define BAUDRATE           115200 // has to be the same as in python
#define Z_LOCKOUT_US 1000

//Variables
volatile long encoder_pos = 0; // GHH60
volatile int8_t dir_from_derivative = 0; // GHH60 
volatile int8_t dir_from_ab = 0; // GHH60 
volatile long z_counter = 0; // GHH60
volatile bool z_primed = false; // GHH60
volatile bool z_armed = true;
volatile unsigned long lastZMicros = 0; // GHH60
volatile float mpc_position = 0; // GHH60
// volatile float absolute_pos = 0;
volatile bool tripped = false; // GHH60

volatile uint16_t zero_raw_top = 0; // AS5600
volatile uint16_t last_raw = 0; // AS5600
volatile float angle_unwrapped = 0; // AS5600

static long last_counts = 0;
static float last_theta = 0.0f;
volatile float alpha_v = 0.2;
volatile float alpha_omega = 0.9;
volatile float v_filtered = 0.0f;
volatile float omega_filtered = 0.0f;
volatile float theta_new = 0.0f; // penelums left = negative angle
volatile bool i2c_request = true;

//velocitys
volatile uint32_t derive_time_us = 5000; // 5 ms
volatile float dt_deriv_ISR = derive_time_us * 1e-6f; // 0.005s
constexpr int omega_window = 6;
volatile float theta_history[omega_window];
volatile uint8_t theta_idx = 0;
volatile bool theta_history_primed = false;
constexpr int v_window = 6;
volatile long counts_history[v_window];
volatile uint8_t counts_idx = 0;
volatile bool counts_history_primed = false;

volatile float mpc_force = 0.0;

//Constants
constexpr long cpr_ghh60 = 1024; // GHH60
constexpr long cpr_as5600 = 4096; // AS5600
constexpr long position_limit = 0.33; // safety isr. currently set to 20cm roughly
constexpr float two_pi = 2.0f * PI; // AS5600
constexpr float four_pi = 4.0f * PI; // AS5600
constexpr float step_rad = (2.0f * PI) / cpr_as5600; // AS5600
constexpr unsigned long communication_time = 10; // has to be the same as in python
constexpr float meters_per_count = 0.04f / (float(cpr_ghh60));


//PID
constexpr int PWM_MAX_CMD   = 100;  // erwarteter Bereich von Python (z. B. [-150..+150])
constexpr int PWM_MIN_EFF   = 20;   // Mindest-PWM, um Reibung zu überwinden
constexpr int PWM_DEADBAND  = 3;    // |u| <= -> Motor aus
constexpr unsigned CMD_TIMEOUT_MS = 100; // wenn länger nichts kommt -> Stop

static int last_pwm_cmd = 0;
static unsigned long last_cmd_ms = 0;

//for testing
static unsigned long test_start_ms = 0;
static bool test_done = false;


//ISR//
////////////////////////////////////////////////////////////////////////
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
  /*g
  -triggered on RISING z edge from GHH60
  -used to reset position to avoid drift
  -with safety mechanism to avoid misscounting
  */

  unsigned long now = micros();
  if ((unsigned long)(now - lastZMicros) < (unsigned long)Z_LOCKOUT_US) {
    return;
  }
  lastZMicros = now;

  #if ZERO_ON_Z_RISING
    encoder_pos = 0;
  #endif

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
    mpc_position = 0;
    z_primed = true;
  }

}


void derivISR() {
  /*
  -derive xdot and thetadot
  -simply derived from postional difference between now and last measurement
  */

  // 1) Cart velocity - GENAU WIE BEI OMEGA!
  long counts = absolute_counts();
  
  counts_idx = (counts_idx + 1) % v_window;
  long counts_old = counts_history[counts_idx];
  counts_history[counts_idx] = counts;

  long diff_counts = counts - counts_old;
  dir_from_derivative = (diff_counts > 0) ? 1 : (diff_counts < 0) ? -1 : 0;
  
  // ✅ JETZT KONSISTENT: Durch Fenstergröße teilen wie bei omega!
  float v_raw = counts_to_meters(diff_counts) / (dt_deriv_ISR * v_window);
  v_filtered = v_raw;

  // pendulum angle velocity thetadot
  float theta = theta_new;

  float theta_old = theta_history[theta_idx]; // ältetse Wert löschen neuen einfügen
  theta_history[theta_idx] = theta;
  theta_idx = (theta_idx + 1) % omega_window;

  float omega_raw;
  if (theta_history_primed) {
    omega_raw = (theta - theta_old) / (dt_deriv_ISR * omega_window);
  }
  else {
    //only relevant for priming. Maybe put this if branch in the setup function later
    omega_raw = (theta - last_theta)/ dt_deriv_ISR;
  }

  last_theta = theta;

  omega_filtered = omega_raw;



  // Bitte bald frischen I2C-Winkel lesen
  i2c_request = true;
}


//Helpers///////////////////////////////////////////////////////////////

inline void setMotorSignedPWM(int u_cmd) {
  int mag = abs(u_cmd);

  if (mag <= PWM_DEADBAND) {        // Deadband: Motor aus
    analogWrite(PWM_PIN, 0);
    return;
  }

  if (mag < PWM_MIN_EFF) mag = PWM_MIN_EFF; // Mindest-PWM
  mag = clampInt(mag, 0, PWM_MAX_CMD);

  if (u_cmd >= 0) {
    // positiver Befehl -> Wagen nach rechts
    digitalWrite(DIR_PIN, LOW);     // bei dir: LOW = rechts
  } else {
    digitalWrite(DIR_PIN, HIGH);    // negativ -> links
  }
  analogWrite(PWM_PIN, mag);
}


inline void applyCommandLine(const char* s) {
  if (tripped) return; // safetyISR
  float uf = atof(s); // robust genug für einfache Zahlen
  // begrenzen auf eingestellten Bereich
  int u = (int)roundf(uf);
  u = clampInt(u, -PWM_MAX_CMD, PWM_MAX_CMD);

  last_pwm_cmd = u;
  last_cmd_ms  = millis();
  setMotorSignedPWM(u);
}

inline int clampInt(int v, int lo, int hi) {
  return (v < lo) ? lo : (v > hi) ? hi : v;
}


long force_to_pwm(float force) {
  /*
  -dummy function hasnt been implementd so far
  */
  
  return 30;
}


long absolute_counts() {
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

float counts_to_meters(long counts) {
  return (float)counts * meters_per_count;
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
  -wraps the absolute postion og the magnetic ecoder to the desired intervall
  */

  a = fmodf(a, four_pi);        // (-4π, 4π)
  if (a <  -two_pi) {
    a += four_pi;
  }
  if (a >   two_pi) {
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
  return diff * step_rad;
}

void init_coordinate_system() {
  digitalWrite(DIR_PIN, LOW);
  analogWrite(PWM_PIN, 40);

  while (z_primed == false) {};
  
  analogWrite(PWM_PIN, 0);
  delay(50);

  return;

}

void motorStop() {
  /*
  -terminates the Motor
  */

  analogWrite(PWM_PIN, 0);
}


///////////////////////////////////////////////////////////////////
void setup() {
  Serial.begin(BAUDRATE);
  while (!Serial) {;}  

  //Magnetic Encoder
  pinMode(20, INPUT_PULLUP); 
  pinMode(21, INPUT_PULLUP); 
  //GHH60 Sensor PINS
  pinMode(PIN_A, INPUT);
  pinMode(PIN_B, INPUT);
  pinMode(PIN_Z, INPUT);

  //Motor PINS
  pinMode(PWM_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);

  //Magnetic Encoder init
  Wire.begin();
  Wire.setClock(400000);

  if (!as5600.begin()) {  //test if the sensor go recognized 
    Serial.println("AS5600 not found on Wire. Check wiring.");
    while (1) { delay(1000); }
  }
  delay(5000);
  uint16_t zero_raw_bottom = readRaw();   

  zero_raw_top = (zero_raw_bottom + (cpr_as5600 / 2)) & 0x0FFF;

  last_raw = readRaw();

  angle_unwrapped = raw_to_rad(last_raw);
  theta_new = angle_unwrapped; // to avoid spike in the beginning
  // Fenster für Ableitung primen
  for (int i = 0; i < omega_window; ++i) {
    theta_history[i] = theta_new;
  }
  theta_idx = 0;
  theta_history_primed = true;

  // priming of velocity buffers
  last_counts = absolute_counts();
  last_theta  = theta_new;

  for (int i = 0; i < v_window; ++i) counts_history[i] = last_counts;
  counts_idx = 0;
  counts_history_primed = true; 


  //Motor init
  digitalWrite(DIR_PIN, HIGH);
  analogWrite(PWM_PIN, 0);

  //Hardware ISR
  attachInterrupt(digitalPinToInterrupt(PIN_A), isrA, RISING);
  attachInterrupt(digitalPinToInterrupt(PIN_Z), isrZ, RISING);

  //init coordinatesystem
  init_coordinate_system();

  //Timer ISR
  //Timer3.attachInterrupt(safetyCheckISR).start(1000);
  Timer4.attachInterrupt(derivISR).start(5000);


  //Pause before start
  delay(5000);

  digitalWrite(DIR_PIN, LOW);     // Richtung festlegen
  test_start_ms = millis();       // Test-Timer starten
  Serial.print("time");
    Serial.print(',');
    Serial.print("x");
    Serial.print(',');
    Serial.println("v");
}


/*
-safety: always in the beginning check if tripped == true, then call motorstop()
-
*/
void loop() {
  // --- PWM-Test: 1 s aus, 1 s 50, dann aus und terminieren ---
  if (!test_done) {
    unsigned long elapsed = millis() - test_start_ms;

    if (elapsed < 1000UL) {
      // 0..1 s: PWM aus
      analogWrite(PWM_PIN, 0);
    } else if (elapsed < 1400UL) {
      // 1..2 s: PWM ~50
      analogWrite(PWM_PIN, 100);;      // deine Skala (0..100)
    } else {
      // >2 s: stoppen & terminieren
      analogWrite(PWM_PIN, 0);
      Serial.println("TEST_DONE");
      test_done = true;
    }
  } else {
    // „terminieren“: hier optional hart anhalten
    while (1) { delay(1000); }
  }

  // --- Telemetrie: x_m und v_filtered kontinuierlich ausgeben ---
  // --- Telemetrie: t_rel [s], x_m [m], v_filtered [m/s] ---
  static unsigned long last_ms = 0;
  static unsigned long t0_ms = 0;         // Startzeit der ersten Messung (relativ)
  unsigned long now = millis();

  if (now - last_ms >= communication_time) {
    last_ms += communication_time;

    if (t0_ms == 0) t0_ms = now;          // beim ersten Mal setzen
    const long SHIFT_MS = 1000;  // wenn du später was anderes willst, hier anpassen
    float t_rel_s = ((long)now - (long)t0_ms - SHIFT_MS) / 1000.0f;

    long  counts = absolute_counts();
    float x_m    = counts_to_meters(counts);

    // Ausgabe: Zeit(s), x(m), v(m/s)
    Serial.print(t_rel_s, 3);  Serial.print(',');
    Serial.print(x_m, 4);      Serial.print(',');
    Serial.println(v_filtered, 4);
  }

}

