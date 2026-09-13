# main.py - Falcon Eye ESP8266 Servo Controller
#
# ESP8266 <-> WiFi <-> MQTT Broker <-> Python Face Recognition
#
# Servo:
#   Signal -> D5 / GPIO14
#   VCC    -> external 5V recommended
#   GND    -> common GND with ESP8266
#
# MQTT commands:
#   falcon/eye/servo/cmd
#
# Commands:
#   ANGLE:0
#   ANGLE:20
#   ANGLE:40
#   ...
#   ANGLE:180
#   STOP
#   HOME
#
# Status:
#   falcon/eye/servo/status

import time
import network
import machine
import ubinascii
import ujson as json
from machine import Pin, PWM
from umqtt.simple import MQTTClient


# ============================================================
# WIFI / MQTT CONFIGURATION
# ============================================================

WIFI_SSID = "RCA"
WIFI_PASS = "@RcaNyabihu2023"

MQTT_HOST = "broker.benax.rw"
MQTT_PORT = 1883

# ============================================================
# MQTT TOPICS
# ============================================================

TOPIC_SERVO_CMD = b"falcon/eye/servo/cmd"
TOPIC_SERVO_STATUS = b"falcon/eye/servo/status"
TOPIC_RECOGNITION = b"falcon/eye/recognition"

CLIENT_ID = b"falcon_eye_" + ubinascii.hexlify(machine.unique_id())
TOPIC_STATUS = b"falcon/eye/status/" + CLIENT_ID


# ============================================================
# SERVO CONFIGURATION
# ============================================================

# D5 = GPIO14 on NodeMCU ESP8266
SERVO_PIN = 14

SERVO_MIN_US = 500
SERVO_MAX_US = 2400

HOME_ANGLE = 0


# ============================================================
# HARDWARE
# ============================================================

servo = PWM(Pin(SERVO_PIN), freq=50)
client = None

current_angle = HOME_ANGLE
servo_stopped = False


# ============================================================
# SERVO FUNCTIONS
# ============================================================

def angle_to_duty_u16(angle):
    """
    Convert 0-180 degrees to PWM duty for a standard servo.
    ESP8266 MicroPython supports duty_u16().
    """

    angle = max(0, min(180, int(angle)))

    pulse_us = (
        SERVO_MIN_US
        + (SERVO_MAX_US - SERVO_MIN_US) * angle / 180
    )

    # 50 Hz = 20,000 us period
    duty = int((pulse_us / 20000.0) * 65535)

    return duty


def move_servo(angle):
    global current_angle
    global servo_stopped

    angle = max(0, min(180, int(angle)))

    duty = angle_to_duty_u16(angle)

    try:
        servo.duty_u16(duty)
    except AttributeError:
        # Compatibility with older MicroPython
        duty10 = int((duty / 65535) * 1023)
        servo.duty(duty10)

    current_angle = angle
    servo_stopped = False

    print("Servo moved to:", angle)


def stop_servo():
    """
    Stop sending active PWM to the servo.

    Note:
    If you want the servo to physically HOLD the position,
    do not disable PWM. The servo will continue receiving its
    last position command.
    """

    global servo_stopped

    # Keep the last PWM position so the servo holds position.
    servo_stopped = True

    print("SERVO STOPPED / HOLDING:", current_angle)


# ============================================================
# MQTT STATUS
# ============================================================

def publish_servo_status(status):
    global client

    if client is None:
        return

    payload = {
        "status": status,
        "angle": current_angle
    }

    try:
        client.publish(
            TOPIC_SERVO_STATUS,
            json.dumps(payload)
        )

        print("Servo status:", payload)

    except Exception as e:
        print("Failed to publish servo status:", e)


# ============================================================
# WIFI
# ============================================================

def wifi_connect():

    print("Connecting to WiFi:", WIFI_SSID)

    sta = network.WLAN(network.STA_IF)

    sta.active(True)

    if not sta.isconnected():

        sta.connect(WIFI_SSID, WIFI_PASS)

        start = time.ticks_ms()

        while not sta.isconnected():

            if time.ticks_diff(
                time.ticks_ms(),
                start
            ) > 20000:

                raise RuntimeError("WiFi timeout")

            time.sleep(0.3)

    print("WiFi connected")
    print("IP configuration:", sta.ifconfig())

    return True


# ============================================================
# MQTT CALLBACK
# ============================================================

def mqtt_callback(topic, msg):

    global current_angle

    try:

        command = msg.decode().strip().upper()

        print()
        print("MQTT command:", command)

        # ----------------------------------------------------
        # ANGLE COMMAND
        # ----------------------------------------------------

        if command.startswith("ANGLE:"):

            value = command.split(":", 1)[1]

            angle = int(value)

            if 0 <= angle <= 180:

                move_servo(angle)

                # Give servo a little time to reach position
                time.sleep_ms(300)

                publish_servo_status(
                    "ANGLE_REACHED"
                )

            else:

                print("Invalid angle:", angle)

                publish_servo_status(
                    "ERROR_INVALID_ANGLE"
                )

        # ----------------------------------------------------
        # STOP COMMAND
        # ----------------------------------------------------

        elif command == "STOP":

            stop_servo()

            publish_servo_status(
                "STOPPED"
            )

        # ----------------------------------------------------
        # HOME COMMAND
        # ----------------------------------------------------

        elif command == "HOME":

            move_servo(HOME_ANGLE)

            time.sleep_ms(300)

            publish_servo_status(
                "HOME"
            )

        else:

            print("Unknown command:", command)

            publish_servo_status(
                "ERROR_UNKNOWN_COMMAND"
            )

    except Exception as e:

        print("Command error:", e)

        try:
            publish_servo_status(
                "ERROR"
            )
        except:
            pass


# ============================================================
# MQTT CONNECTION
# ============================================================

def mqtt_connect():

    print("Connecting to MQTT:", MQTT_HOST)

    c = MQTTClient(
        client_id=CLIENT_ID,
        server=MQTT_HOST,
        port=MQTT_PORT,
        keepalive=30
    )

    c.set_last_will(
        TOPIC_STATUS,
        b"offline",
        retain=True
    )

    c.connect()

    c.publish(
        TOPIC_STATUS,
        b"online",
        retain=True
    )

    c.set_callback(mqtt_callback)

    c.subscribe(TOPIC_SERVO_CMD)

    print(
        "MQTT connected as:",
        CLIENT_ID.decode()
    )

    print(
        "Subscribed:",
        TOPIC_SERVO_CMD.decode()
    )

    return c


# ============================================================
# BOOT
# ============================================================

print("=" * 60)
print("FALCON EYE ESP8266 SERVO CONTROLLER")
print("=" * 60)

try:

    # Start servo at home position
    move_servo(HOME_ANGLE)

    wifi_connect()

    client = mqtt_connect()

    publish_servo_status(
        "READY"
    )

    print("=" * 60)
    print("FALCON EYE READY")
    print("=" * 60)

except Exception as e:

    print("BOOT FAILED:", e)

    try:
        servo.deinit()
    except:
        pass

    raise


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    try:

        client.check_msg()

        time.sleep_ms(50)

    except KeyboardInterrupt:

        print()
        print("Shutting down...")

        try:
            client.publish(
                TOPIC_STATUS,
                b"offline",
                retain=True
            )

            client.disconnect()

        except:
            pass

        try:
            servo.deinit()
        except:
            pass

        break

    except Exception as e:

        print("MQTT error:", e)

        time.sleep(2)

        try:

            client = mqtt_connect()

            publish_servo_status(
                "RECONNECTED"
            )

        except Exception as reconnect_error:

            print(
                "Reconnect failed:",
                reconnect_error
            )

            time.sleep(5)