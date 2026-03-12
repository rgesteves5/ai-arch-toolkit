Date & Time — _datetime.py

- datetime_now — Current date/time in a given timezone
- timezone_convert — Convert time between timezones
- date_add — Add days/hours/minutes to a date or datetime
- date_diff — Difference between two dates/datetimes in a chosen unit
- date_format — Reformat a date/datetime using strftime syntax

Weather — _weather.py

- get_weather — Current weather for a city (temp, humidity, wind, conditions)
- get_forecast — Multi-day weather forecast for a city
- get_weather_by_coords — Current weather for a latitude/longitude pair
- get_forecast_by_coords — Multi-day forecast for a latitude/longitude pair
- weather_units — Current weather for a city with Celsius/Fahrenheit output

Geography — _geo.py

- geocode — Coordinates and country for a city name
- reverse_geocode — Place name and region from coordinates
- timezone_lookup — Timezone and UTC offset from coordinates
- distance_between — Great-circle distance between coordinate pairs
- ip_lookup — Geographic location and ISP info for an IP
- country_info — Country details (capital, population, languages, etc.)

Wikipedia — _wikipedia.py

- wikipedia_search — Search Wikipedia, return titles with summaries
- wikipedia_article — Get a specific Wikipedia article summary
- wikipedia_related — Get related article titles from a Wikipedia page

Dictionary — _dictionary.py

- define_word — Dictionary definition via Free Dictionary API

News — _news.py

- hacker_news — Top stories from Hacker News

Math — _math.py

- math_eval — Safely evaluate math expressions (functions, constants, operators)
- unit_convert — Convert between units (length, mass, volume, speed, area, time, temp)

Text Processing — _text.py

- regex_search — Find all regex matches with positions
- text_stats — Count words, characters, lines, sentences, paragraphs
- base64_encode — Encode text to base64
- base64_decode — Decode base64 to text

Data — _json.py

- json_extract — Extract values from JSON via dot-notation paths
- csv_read — Read CSV files, return formatted table

Filesystem — _filesystem.py

- read_file — Read file contents with optional line limit
- list_directory — List files/dirs with sizes and types
- search_files — Recursively search for text in files

Shell — _shell.py

- run_command — Execute shell commands and return output

Web — _web.py

- http_get — Fetch a URL, return raw response text
- scrape_text — Fetch web page, extract visible text (strips HTML)
