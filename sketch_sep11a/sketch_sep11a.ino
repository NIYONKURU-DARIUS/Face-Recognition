// FalconEye_ServoController.ino  (ESP32 version)
// ESP32 <-> WiFi <-> MQTT Broker <-> Python Face Recognition

#include <WiFi.h>
#include <PubSubClient.h>
#include <ESP32Servo.h>

// ============================================================
// WIFI / MQTT CONFIGURATION
// ============================================================
const char* WIFI_SSID = "Y3C";
const char* WIFI_PASS = "RCA@2024";

const char* MQTT_HOST = "broker.benax.rw";
const int   MQTT_PORT = 1883;

// ============================================================
// MQTT TOPICS
// ============================================================
const char* TOPIC_SERVO_CMD    = "falcon/eye/servo/cmd";
const char* TOPIC_SERVO_STATUS = "falcon/eye/servo/status";

String clientId;
String topicStatus;

// ============================================================
// SERVO CONFIGURATION
// ============================================================
const int SERVO_PIN = 18;   // GPIO18
const int SERVO_MIN_US = 500;
const int SERVO_MAX_US = 2400;
const int HOME_ANGLE = 90;   // start / home position = center

Servo servo;
int currentAngle = HOME_ANGLE;
bool servoStopped = false;

WiFiClient espClient;
PubSubClient client(espClient);

// ============================================================
// SERVO FUNCTIONS
// ============================================================
void moveServo(int angle) {
  angle = constrain(angle, 0, 180);
  servo.write(angle);
  currentAngle = angle;
  servoStopped = false;
  Serial.print("Servo moved to: ");
  Serial.println(angle);
}

void stopServo() {
  servoStopped = true;
  Serial.print("SERVO STOPPED / HOLDING: ");
  Serial.println(currentAngle);
}

// ============================================================
// MQTT STATUS
// ============================================================
void publishServoStatus(const char* status) {
  String payload = String("{\"status\":\"") + status +
                    "\",\"angle\":" + currentAngle + "}";
  client.publish(TOPIC_SERVO_STATUS, payload.c_str());
  Serial.print("Servo status: ");
  Serial.println(payload);
}

// ============================================================
// WIFI
// ============================================================
void wifiConnect() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASS);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - start > 20000) {
      Serial.println("WiFi timeout");
      ESP.restart();
    }
    delay(300);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

// ============================================================
// MQTT CALLBACK
// ============================================================
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String command;
  for (unsigned int i = 0; i < length; i++) {
    command += (char)payload[i];
  }
  command.trim();
  command.toUpperCase();

  Serial.println();
  Serial.print("MQTT command: ");
  Serial.println(command);

  if (command.startsWith("ANGLE:")) {
    int angle = command.substring(6).toInt();
    if (angle >= 0 && angle <= 180) {
      moveServo(angle);
      delay(300);
      publishServoStatus("ANGLE_REACHED");
    } else {
      Serial.println("Invalid angle");
      publishServoStatus("ERROR_INVALID_ANGLE");
    }
  }
  else if (command == "STOP") {
    stopServo();
    publishServoStatus("STOPPED");
  }
  else if (command == "HOME") {
    moveServo(HOME_ANGLE);
    delay(300);
    publishServoStatus("HOME");
  }
  else {
    Serial.println("Unknown command");
    publishServoStatus("ERROR_UNKNOWN_COMMAND");
  }
}

// ============================================================
// MQTT CONNECTION
// ============================================================
void mqttConnect() {
  client.setServer(MQTT_HOST, MQTT_PORT);
  client.setCallback(mqttCallback);

  while (!client.connected()) {
    Serial.print("Connecting to MQTT: ");
    Serial.println(MQTT_HOST);

    if (client.connect(clientId.c_str(),
                        topicStatus.c_str(), 0, true, "offline")) {
      Serial.print("MQTT connected as: ");
      Serial.println(clientId);

      client.publish(topicStatus.c_str(), "online", true);
      client.subscribe(TOPIC_SERVO_CMD);

      Serial.print("Subscribed: ");
      Serial.println(TOPIC_SERVO_CMD);
    } else {
      Serial.print("MQTT connect failed, rc=");
      Serial.println(client.state());
      delay(2000);
    }
  }
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("============================================================");
  Serial.println("FALCON EYE ESP32 SERVO CONTROLLER");
  Serial.println("============================================================");

  // Servo setup
  servo.setPeriodHertz(50);
  servo.attach(SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
  moveServo(HOME_ANGLE);

  // Unique client ID from MAC address
  uint64_t chipid = ESP.getEfuseMac();
  char idBuf[13];
  snprintf(idBuf, sizeof(idBuf), "%012llX", chipid);
  clientId = String("falcon_eye_") + idBuf;
  topicStatus = String("falcon/eye/status/") + clientId;

  wifiConnect();
  mqttConnect();

  publishServoStatus("READY");

  Serial.println("============================================================");
  Serial.println("FALCON EYE READY");
  Serial.println("============================================================");
}

// ============================================================
// MAIN LOOP
// ============================================================
void loop() {
  if (!client.connected()) {
    Serial.println("MQTT disconnected, reconnecting...");
    mqttConnect();
    publishServoStatus("RECONNECTED");
  }

  client.loop();
  delay(50);
}