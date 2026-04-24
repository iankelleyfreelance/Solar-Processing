import matplotlib.pyplot
import numpy
import datetime

dates = [datetime.datetime(2026, 1, 26, 17, 21, 57),
datetime.datetime(2026, 2, 6, 20, 7, 35),
datetime.datetime(2026, 2, 7, 18, 53, 47),
datetime.datetime(2026, 2, 9, 18, 1, 37),
datetime.datetime(2026, 2, 12, 17, 32, 30),
datetime.datetime(2026, 2, 16, 17, 7, 2),
datetime.datetime(2026, 2, 16, 17, 7, 2),
datetime.datetime(2026, 2, 16, 17, 30, 20),
datetime.datetime(2026, 2, 16, 17, 35, 26),
datetime.datetime(2026, 2, 16, 17, 58, 45),
datetime.datetime(2026, 2, 20, 16, 0, 57),
datetime.datetime(2026, 2, 22, 0, 0, 0),
datetime.datetime(2026, 2, 23, 0, 0, 0),
datetime.datetime(2026, 2, 27, 18, 56, 30),
datetime.datetime(2026, 2, 28, 18, 38, 25),
datetime.datetime(2026, 3, 17, 22, 5, 16),
datetime.datetime(2026, 3, 22, 19, 31, 41),
datetime.datetime(2026, 3, 23, 20, 16, 51),
datetime.datetime(2026, 3, 24, 16, 56, 55)
]


plot = matplotlib.pyplot.bar(dates, [1]*len(dates))
matplotlib.pyplot.xlabel("Date")
matplotlib.pyplot.ylabel("Image Number")
matplotlib.pyplot.title("H-alpha Solar Images")

matplotlib.pyplot.show()