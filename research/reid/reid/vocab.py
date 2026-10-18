"""Generator-side vocabulary (independent of MedAnon's lexicons: nothing here is read from src/medanon/resources).

Occupations carry an education level and a shift-work flag only; the attacker maps titles to MedAnon's public
occupation categories on its own side. All names, places and people produced with this vocabulary are fictional.
"""

# title, education ('higher' | 'hs'), shift work possible
OCCUPATIONS = [
    ("software engineer", "higher", 0), ("accountant", "higher", 0), ("high school teacher", "higher", 0),
    ("elementary school teacher", "higher", 0), ("registered nurse", "higher", 1), ("pharmacist", "higher", 0),
    ("civil engineer", "higher", 0), ("lawyer", "higher", 0), ("architect", "higher", 0),
    ("physical therapist", "higher", 0), ("financial analyst", "higher", 0), ("marketing manager", "higher", 0),
    ("graphic designer", "higher", 0), ("social worker", "higher", 0), ("librarian", "higher", 0),
    ("dentist", "higher", 0), ("veterinarian", "higher", 0), ("research scientist", "higher", 0),
    ("college professor", "higher", 0), ("journalist", "higher", 0), ("human resources manager", "higher", 0),
    ("data analyst", "higher", 0), ("project manager", "higher", 0), ("physician assistant", "higher", 1),
    ("speech therapist", "higher", 0), ("occupational therapist", "higher", 0), ("real estate agent", "higher", 0),
    ("insurance underwriter", "higher", 0), ("urban planner", "higher", 0), ("sales manager", "higher", 0),
    ("truck driver", "hs", 1), ("bus driver", "hs", 1), ("electrician", "hs", 0), ("plumber", "hs", 0),
    ("carpenter", "hs", 0), ("construction worker", "hs", 0), ("warehouse worker", "hs", 1), ("cashier", "hs", 0),
    ("retail sales associate", "hs", 0), ("waiter", "hs", 1), ("line cook", "hs", 1), ("barista", "hs", 0),
    ("hairdresser", "hs", 0), ("security guard", "hs", 1), ("auto mechanic", "hs", 0), ("delivery driver", "hs", 0),
    ("janitor", "hs", 1), ("home health aide", "hs", 1), ("nursing assistant", "hs", 1), ("receptionist", "hs", 0),
    ("bank teller", "hs", 0), ("landscaper", "hs", 0), ("welder", "hs", 0), ("roofer", "hs", 0),
    ("factory worker", "hs", 1), ("forklift operator", "hs", 1), ("firefighter", "hs", 1), ("police officer", "hs", 1),
    ("mail carrier", "hs", 0), ("childcare worker", "hs", 0), ("housekeeper", "hs", 0), ("house painter", "hs", 0),
    ("machinist", "hs", 0), ("dental assistant", "hs", 0), ("call center representative", "hs", 1),
    ("farm worker", "hs", 0), ("commercial fisherman", "hs", 0), ("flight attendant", "hs", 1), ("taxi driver", "hs", 1),
    ("bartender", "hs", 1),
]
HOBBIES = ["running", "marathon running", "cycling", "swimming", "hiking", "gardening", "tennis", "golf", "yoga",
           "playing guitar", "knitting", "fishing", "hunting", "woodworking", "painting", "chess", "coaching soccer",
           "rock climbing", "bird watching", "photography", "baking", "salsa dancing", "skiing",
           "volunteering at a food bank", "singing in a choir", "bowling", "kayaking", "weightlifting"]
TRAVEL = ["mexico", "canada", "italy", "india", "brazil", "the dominican republic", "portugal", "ireland", "china",
          "kenya", "vietnam", "germany", "jamaica", "peru", "the philippines", "egypt", "thailand", "greece"]
# (dialogue phrase, note phrase) per work pattern
WORK_PATTERNS = {
    "night shifts": ("i work nights", "works night shifts"),
    "rotating shifts": ("my shifts rotate a lot", "works rotating shifts"),
    "remote": ("i work from home most days", "works remotely"),
    "long hauls": ("i'm on the road for days at a time", "is away on multi-day routes"),
    "part-time": ("i only work part time", "works part-time"),
}
MARITAL_WORDS = {"M": ("married", ["my husband", "my wife"]), "D": ("divorced", ["my ex"]),
                 "W": ("widowed", ["my late husband", "my late wife"]), "S": ("single", ["my partner"])}
