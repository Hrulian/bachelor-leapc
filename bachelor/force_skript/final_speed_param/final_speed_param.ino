
#include <DueTimer.h>


//Motor
const int PWM_PIN = 5;
const int DIR_PIN = 6; // LOW is to the left currently

// GHH60
constexpr uint8_t PIN_A = 2;
constexpr uint8_t PIN_B = 3;

constexpr long cpr_ghh60 = 1024; 
constexpr float TAU_V = 0.01f;  // 10 ms
constexpr uint32_t position_limit = 13000;

volatile bool tripped = false; // flag that indicates whether we exceeded the position limit
volatile long x = 0; // cartpostion in coutns
volatile long v = 0; // cartposition in counts/s
volatile long encoder_pos = 0; 

// increase baudrate for high-rate telemetry (set same on host)
constexpr uint32_t BAUDRATE = 115200;
constexpr uint32_t STATE_UPDATE_US = 1000;
// --- User-configurable test params ---
constexpr int APPLY_PWM_CMD = -138; // signed PWM command applied when streaming starts
// request 2 ms telemetry interval
constexpr unsigned long TELEMETRY_MS = 2; // telemetry interval in ms

// derived constants
constexpr unsigned long communication_time = TELEMETRY_MS; // ms between telemetry samples
constexpr float meters_per_count = 0.04f / float(cpr_ghh60);

// test flags
static unsigned long test_start_ms = 0;
static bool test_done = false;

// --- konfigurieren ---
constexpr unsigned long SAMPLE_US = 2000;      // 2 ms
constexpr int BUF_SIZE = 512;                   // ringbuffer size (power of two recommended)
volatile unsigned long sample_t_us_buf[BUF_SIZE];
volatile long sample_counts_buf[BUF_SIZE];
volatile uint16_t buf_head = 0; // next write index (ISR)
volatile uint16_t buf_tail = 0; // next read index (loop)

// pwm start time in microseconds
volatile unsigned long pwm_start_us = 0;
volatile bool pwm_applied = false;

// ISR: liefere sample in ringbuffer
void sampleISR() {
  if (!pwm_applied) return;               // nur samplen, wenn PWM aktiv
  unsigned long t = micros();             // exakter Zeitpunkt
  long cnt;
  // sicherer Lesezugriff auf encoder in ISR
  noInterrupts();
  cnt = encoder_pos;
  interrupts();

  uint16_t next = (buf_head + 1) & (BUF_SIZE - 1);
  if (next != buf_tail) {                 // Buffer nicht voll
    sample_t_us_buf[buf_head] = t;
    sample_counts_buf[buf_head] = cnt;
    buf_head = next;
  } else {
    // Buffer voll -> Sample verwerfen oder handle overflow (increment tail to drop oldest)
    buf_tail = (buf_tail + 1) & (BUF_SIZE - 1);
    sample_t_us_buf[buf_head] = t;
    sample_counts_buf[buf_head] = cnt;
    buf_head = next;
  }
}



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



//HELPERS------------------------------------------------------------------------------


void motorstop() {
  /*
  -terminates the Motor
  -used currently only for safety watchdog
  */

  analogWrite(PWM_PIN, 0);
}


void setup() {
  // begin serial communication
  Serial.begin(BAUDRATE);
  while (!Serial) {}
  
  // Motor PINS
  pinMode(PWM_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);

  
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
  delay(5000);

  test_start_ms = millis();       // Test-Timer starten
  // print CSV header (pwm, t in ms relative to PWM start, encoder counts)
  Serial.println("pwm,t_ms,x");

  // stelle Timer ein (startet, aber ISR macht nur was wenn pwm_applied == true)
  Timer4.attachInterrupt(sampleISR).start(SAMPLE_US); // oder Timer3/Timer4 je nach Verfügbarkeit
}


void loop() {
  // If safety tripped already -> ensure motor stopped and do nothing
  if (abs(encoder_pos) > position_limit) {
    tripped = true;
  }

  if (tripped) {
    motorstop();
    // stay here - safety tripped
    while (1) { delay(1000); }
  }

  // Apply configured PWM command (signed) directly once: set direction by sign and magnitude
  // Use the global volatile `pwm_applied` and `pwm_start_us` so ISR and loop see the same state
  static int applied_pwm_local = APPLY_PWM_CMD;
  if (!pwm_applied && !tripped) {
    encoder_pos = 0;
    int mag = abs(applied_pwm_local);
    mag = constrain(mag, 0, 255);
    if (applied_pwm_local >= 0) digitalWrite(DIR_PIN, LOW);
    else digitalWrite(DIR_PIN, HIGH);
    analogWrite(PWM_PIN, mag);
    // mark global flags used by ISR and buffer reader
    pwm_applied = true;
    pwm_start_us = micros();
    // optional: clear buffer
    noInterrupts();
    buf_head = buf_tail = 0;
    interrupts();

    // first data line at t=0 (ms)
    char buf0[64];
    int l0 = snprintf(buf0, sizeof(buf0), "%d,%lu,%ld\n", applied_pwm_local, 0UL, encoder_pos);
    Serial.write((const uint8_t*)buf0, l0);
  }


  // --- Telemetry: print time(s), x(m), v(m/s) at `communication_time` interval ---
  static unsigned long last_ms = 0;
  static unsigned long t0_ms = 0; // start time for relative t
  unsigned long now = millis();

  // We rely on the timer ISR filling the ring buffer with samples at SAMPLE_US.
  // The buffered samples are flushed below; no separate millis()-based sampling here.

  // Buffer auslesen und auf Serial schreiben
  while (buf_tail != buf_head) {
    // kritische Sektion: holen eines Eintrags
    noInterrupts();
    uint16_t idx = buf_tail;
    buf_tail = (buf_tail + 1) & (BUF_SIZE - 1);
    unsigned long t_us = sample_t_us_buf[idx];
    long counts = sample_counts_buf[idx];
    interrupts();

    // berechne ms relativ zum PWM-Start
    unsigned long t_ms = (t_us - pwm_start_us) / 1000UL;
    // formatiere CSV: pwm,t_ms,x
    char buf[64];
    int len = snprintf(buf, sizeof(buf), "%d,%lu,%ld\n", APPLY_PWM_CMD, t_ms, counts);
    Serial.write((const uint8_t*)buf, len);
  }
}

