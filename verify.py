from recognize import text


num_plates = {
    "stolen": ["MY70 BMW", "HR 26.BR 9044,"],
    "unregistered": ["MH13AZ9456"]
}

if text in num_plates["stolen"]:
    print()
    print("STATUS: STOLEN VEHICLE!" + "\n")
elif text in num_plates["unregistered"]:
    print()
    print("STATUS: UNREGISTERED VEHICLE!" + "\n")
