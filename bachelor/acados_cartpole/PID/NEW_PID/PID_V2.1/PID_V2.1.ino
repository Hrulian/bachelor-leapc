#include <Wire.h>
#include <AS5600.h>

//AS5600
AS5600 as5600(&Wire);

//Motor
const int PWM_PIN = 5;
const int DIR_PIN = 6; // LOW is to the left currently

// GHH60
constexpr uint8_t PIN_A = 2;
constexpr uint8_t PIN_B = 3;
constexpr uint8_t PIN_Z = 4;

volatile int8_t dir_from_ab = 0; // GHH60 
volatile long encoder_pos = 0; // GHH60

//AS5600
constexpr long cpr_as5600 = 4096;
constexpr float two_pi = 2.0f * PI;
constexpr float four_pi = 4.0f * PI;
constexpr float step_rad = (2.0f * PI) / cpr_as5600;
constexpr uint32_t ANGLE_PERIOD_US = 1000;
constexpr float TAU_S = 0.005; 

static float last_theta = 0.0f;
volatile float theta_new = 0.0f;
volatile uint16_t zero_raw_top = 0;
volatile uint16_t last_raw = 0;
volatile float angle_unwrapped = 0;
volatile float omega = 0;

// tets variables
volatile float x = 11.11;
volatile float v = 22.22;
volatile uint16_t broken_frame = 0; 

//communication
constexpr long BAUDRATE = 115200;
constexpr unsigned long COMMUNICATION_TIME_MS = 5;
constexpr byte NUM_CHARS = 32;

volatile bool new_data = false; // if u has been send to motor -> false
volatile char receivedChars[NUM_CHARS];

volatile float u = 0.0f;


// CHATGPT
// --- Motor-Output Tuning (füge das zu deinen Konstanten) ---
constexpr int U_MAX        = 150;  
constexpr int PWM_DEADBAND = 3;    // <3 -> Motor aus
constexpr int PWM_MIN_EFF  = 20;   // Mindest-PWM, um Reibung zu überwinden

// merkt sich die letzte Richtung, um Deadtime nur bei Richtungswechsel zu setzen
static int last_sign = 0;



// ISR
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


//Communication-------------------------------------------------------------------------
void receive_parse_data() {
  /*
  -parses and receives in the form: <float>
  */

  static bool in_progress = false;
  static byte idx = 0;
  
  while (Serial.available() > 0) {
    char rc = Serial.read();

    if (in_progress == true) {
      if (rc == '>') {
        // frame finished -> reset everything and terminate
        receivedChars[idx] = '\0';
        idx = 0;
        in_progress = false;

        // set the u command and new_data = true
        u = atof((const char*)receivedChars);
        new_data = true;
      }
      else {
        if (rc == '\r' || rc == '\n') continue; // handle errors
        if (idx + 1 < NUM_CHARS) {
          // we currently read the payload
          receivedChars[idx] = rc;
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


void send_state(float x, float v, float theta, float omega) {
  /*
  -sends a state as a string in the form 
  */

  float t = millis() * 0.001f; // in seconds

  Serial.write('<'); // startmarker
  Serial.print(encoder_pos);         
  Serial.write(',');
  Serial.print(v, 3);
  Serial.write(',');
  Serial.print(theta, 3);
  Serial.write(',');
  Serial.print(omega, 3);
  Serial.write('>'); // endmarker
  Serial.write('\n'); // only for debugging  
}



//HELPERS--------------------------------------------------------------------------------

inline void update_theta_omega(const float T)
{
  // 1) I²C lesen
  const uint16_t raw = readRaw();
  const int16_t  dr  = delta_from_raw(raw, last_raw);
  last_raw           = raw;

  // 2) Winkel fortschreiben (unwrapped)
  const float dtheta = (float)dr * step_rad;
  angle_unwrapped   += dtheta;

  // 3) Roh-ω
  float omega_raw = dtheta / T; // rad/s

 
  static float omega_hat = 0.0f;
  const float alpha = T / (TAU_S + T);

  omega_hat = (1.0f - alpha) * omega_hat + alpha * omega_raw;
  omega = omega_hat;
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


void apply_u() {
  new_data = false;

  int u_cmd = (int)lroundf(u);
  u_cmd = constrain(u_cmd, -U_MAX, U_MAX);

  int sign = (u_cmd >= 0) ? +1 : -1;
  int mag  = abs(u_cmd);

  if (mag < PWM_DEADBAND) {
    analogWrite(PWM_PIN, 0);
    return;
  }
  if (mag < PWM_MIN_EFF) {
    mag = PWM_MIN_EFF;
  }

  if (sign != last_sign) {
    analogWrite(PWM_PIN, 0);
    delayMicroseconds(50);
    digitalWrite(DIR_PIN, (sign > 0) ? LOW : HIGH);  // bei dir: LOW = rechts
    delayMicroseconds(50);
    last_sign = sign;
  } else {
    // Richtung bleibt gleich
    digitalWrite(DIR_PIN, (sign > 0) ? LOW : HIGH);
  }

  analogWrite(PWM_PIN, mag);
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
  Wire.setClock(400000);

  
  if (!as5600.begin()) {  //test if the sensor go recognized 
    Serial.println("AS5600 not found on Wire. Check wiring.");
    while (1) { delay(1000); }
  }

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
  pinMode(PIN_Z, INPUT);

  // ISR
  attachInterrupt(digitalPinToInterrupt(PIN_A), isrA, RISING);

  // to show that we are ready
  digitalWrite(DIR_PIN, LOW);
  analogWrite(PWM_PIN, 40);
  delay(100);
  analogWrite(PWM_PIN, 0);


  // pause before start
  delay(5000);
}


void loop() {


  receive_parse_data();

  if (new_data) {
    // new frame arrived and has been added to the buffer
    apply_u();
  }
  
  // --- Zeitraster für Sensor-Update (hier: 1 ms) ---
  static uint32_t time_last_angle = 0;
  uint32_t nowu = micros();

  if ((uint32_t)(nowu - time_last_angle) >= ANGLE_PERIOD_US) {
    // Überfällige Perioden nachholen (robust gegen Jitter)
    uint32_t periods = 0;
    do { time_last_angle += ANGLE_PERIOD_US; periods++; }
    while ((uint32_t)(nowu - time_last_angle) >= ANGLE_PERIOD_US);

    // echtes Δt in s
    const float T = periods * (ANGLE_PERIOD_US * 1e-6f);

    // -> hier nur noch die fachliche Schätzung (I²C + Ableitung)
    update_theta_omega(T);

    // (Später kannst du hier auch Wagen-Geschwindigkeit etc. mit demselben T updaten.)
  }


  // main 10ms controll loop
  static unsigned long next_ms = 0;
  unsigned long now = millis();
  if ((long)(now - next_ms) >= 0) {
    do { next_ms += COMMUNICATION_TIME_MS; } while ((long)(now - next_ms) >= 0);


    float theta = wrap_to_pi(angle_unwrapped);
    //send the state here 
    send_state(x, v, theta, omega);

  }
}