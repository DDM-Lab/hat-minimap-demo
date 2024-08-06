# Interactive Web-based HAT Minimap (Demo)
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
    - Python 3.9 installed [note that it **must** be exactly this version, a later version of Python will **not** work]
    - A Web browser
    - [and MySQL!]

## Local Installation
1. In a command shell, goto the main folder of the cloned git repository which contains the `requirements.txt` file.
2. (suggestion) Create a virtual Python Environment by running the following commands in your shell. (This may be different on your machine!  If it does not work, look at how to install a virtual python env based on your system.):
    - `python3 -m venv env`
    - `source env/bin/activate`
3. Install the required python libraries by running this command in your shell:
    - `pip install -r requirements.txt`
    - [several of the dependencies originally specified for this are no longer supported in PyPi and have been updated to more recent ones]
    - [**before** doing the requirements.txt install, install `wheel` as well]
    - [it is worth noting that among the dependencies is a forked version of python-socketio which is maintained in Ngoc’s private GitHub space; I don’t know what this was forked]
4. Create database named `team_minimap_bot` [you will also need to tinker with permissions to make sure the code can access this database! See mission/db.py for what it needs to be.]
5. Run the file `script.sql` to create according schemas for the `team_minimap_bot` db.
6. [Ensure that the port this wants to use is open; the default choice has been changed to avoid a conflict on Janus]

## Notice
This application is undergoing testing and has not yet been finished. Any suggestions and code updates/requests are welcome.
+ In case of running into an issue "Too many packets in payload", increase the value *"number of max_decode_packets"* in the file `env/lib/python3.x/site-packages/engineio/payload.py`. 
+ To modify the database configuration, check the file `mission/db.py`

## Testing
Use the following URL structure to run the demo.
`http://0.0.0.0:5704/fullmap/<arbitrarily_chosen_username>`



