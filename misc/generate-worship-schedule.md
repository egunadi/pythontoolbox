- have user supply a csv file with names and availability for the week

| name  | week1 | week2 | week3 | week4 |
|-------|-------|-------|-------|-------|
| hezky | True  | True  | False | True  |
| steph | False | True  | False | True  |
| wawan | True  | True  | True  | True  |

- have user supply JSON file with list roles and assigned names

```json
{
  "speaker": ["hezky", "steph"],
  "guitarist": ["wawan"]
}
```

- optional: pass a list of "priority" names

- generate csv file(s) with potential schedule for the month

- implementation ideas
  - create a class for each user to encapsulate each user's rules
  - create a function to scrub the availability table
    - ex. if "steph" cannot serve alongside "hezky", replace her True with False on his weeks
    - this may result in multiple availability tables and possible schedules
  - create a function to fill in slots for a given week's schedule
    - weeks 2-4 can check schedule from prior weeks
    - if "prority" names are given, prioritize those first
      - otherwise, shuffle the selection
  - a master function can instantiate all the users and assign them their roles
    - afterwards, it can recursively generate every week's schedule
    - branches where the full month cannot be scheduled are discarded
