# Write excite_inp.txt
# Edit EWIN_LOW, EWIN_HIGH, and SEED for your system.

EWIN_LOW  = 1.6    # lower bound of excitation window, eV
EWIN_HIGH = 3.3    # upper bound of excitation window, eV
SEED      = 1234   # random seed; use the string '!' to seed from system time

excite_inp = f"""\n\n\n{EWIN_LOW} {EWIN_HIGH}\n\n{SEED}\n\n"""

with open("excite_inp.txt", "w") as f:
    f.write(excite_inp)

print("excite_inp.txt written. Run excite.py with:")
print("  python excite.py < excite_inp.txt")

# then from bash run: python excite.py < excite_inp.txt