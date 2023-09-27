# Interactive Web-based Team Minimap (Demo)
Demo version of 2D browser-based Team Minimap Search and Rescue mission. It was written in Python and Javascript and uses the FastAPI Python web framework. 

- Navigate the grid: `Arrow Keys` or `Arrow Keys + X` to speed up the move
- Open a door: `Enter` (only Engineer)
- Clear rubble: `Enter` x 5 times (only Engineer)
- Rescue GREEN victims: `Enter` x 10 times
- Rescue YELLOW victims: `Enter` x 20 times; Engineer must clear rubble first, then the Medic saves the Yellow victim
- Rescue RED victims: `Enter` x 20 times requiring the presence of Medic and Engineer
- Yellow victims disappear after 4 minutes

## Requirements:
- To run locally:
    - Python 3.9 installed
    - A Web browser

## Local Installation
1. In a command shell, goto the main folder of the cloned git repository which contains the `requirements.txt` file.
2. (suggestion) Create a virtual Python Environment by running the following commands in your shell. (This may be different on your machine!  If it does not work, look at how to install a virtual python env based on your system.):
    - `python3 -m venv env`
    - `source env/bin/activate`
3. Install the required python libraries by running this command in your shell:
    - `pip install -r requirements.txt`
4. Create database named `team_minimap_bot`
5. Run the file `script.sql` to create according schemas for the `team_minimap_bot` db. 

## Notice
This application is undergoing testing and has not yet been finished. Any suggestions and code updates/requests are welcome.
+ In case of running into an issue "Too many packets in payload", increase the value *"number of max_decode_packets"* in the file `env/lib/python3.x/site-packages/engineio/payload.py`. 
+ To modify the database configuration, check the file `mission/db.py`

## Testing
Use the following URL structure to run the demo.
`http://0.0.0.0:5704/fullmap/<arbitrarily_chosen_username>`



