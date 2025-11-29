# audio_devices.py
import sounddevice as sd

def list_all_devices():
    print(sd.query_devices())                     # pretty table
    print("\nDetailed list:")
    for i, dev in enumerate(sd.query_devices()):
        print(f"{i:2d}: {dev['name']:50s} → {dev['max_input_channels']} in, {dev['max_output_channels']} out")

def find_vb_cable_devices():
    devices = sd.query_devices()
    cable_input = None   # we will SEND audio here (virtual output device)
    cable_output = None  # we will RECORD from here (virtual input device)

    for i, dev in enumerate(devices):
        name = dev['name']
        if "CABLE Output" in name and dev['max_input_channels'] > 0:
            cable_output = i
        elif "Output (VB-Audio Point)" in name and dev['max_output_channels'] > 0:
            cable_input = i

    return cable_input, cable_output

if __name__ == "__main__":
    list_all_devices()

    send_to_cable_device, receive_from_cable_device = find_vb_cable_devices()

    print("\nVB-Cable setup on your system:")
    print(f"  • Send audio INTO the cable  (output device)  → index {send_to_cable_device}")
    print(f"  • Receive audio FROM the cable (input device) → index {receive_from_cable_device}")

    # Example: set as default for the current Python process only
    if send_to_cable_device is not None and receive_from_cable_device is not None:
        sd.default.device = (receive_from_cable_device, send_to_cable_device)  # (input, output)
        print("\nsounddevice defaults updated for this script only:")
        print("   sd.default.device =", sd.default.device)