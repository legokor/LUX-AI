import math, sys
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

DIRECTIONS = Constants.DIRECTIONS
game_state = None

def get_input(observation, game_state, unit):
    """
    A függvény visszadja a neurális hálózat bemenetéhez szükséges adatokat
    - Legközelebbi nyersanyag távolsága
    - Worker maradék helye
    - Legközelebbi város távolsága
    - Legközelebbi város üzemanyaga
    - Legközelebbi város üzemanyag igénye
    - Est-Nap ciklus 
    - Tud-e építeni a worker
    """

    # Closest resource 
    player = game_state.players[observation.player]
    width, height = game_state.map.width, game_state.map.height

    resource_tiles: list[Cell] = []
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)
    
    closest_resource_dist = math.inf
    closest_resource_tile = None

    for resource_tile in resource_tiles:
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        dist = resource_tile.pos.distance_to(unit.pos)
        if dist < closest_resource_dist:
            closest_resource_dist = dist
            closest_resource_tile = resource_tile
    
    # Space left
    space_left = unit.get_cargo_space_left()

    # Closest city
    closest_city_dist = math.inf
    closest_city_tile = None
    for k, city in player.cities.items():
        for city_tile in city.citytiles:
            dist = city_tile.pos.distance_to(unit.pos)
            if dist < closest_city_dist:
                closest_city_dist = dist
                closest_city_tile = city_tile
                # Closest city fuel
                closest_city_fuel = city.fuel
                # Closest city light_upkeep
                closest_city_light_upkeep = city.light_upkeep
    
    # Day Night cycle
    cycle_position = observation["step"] % 40

    # Can the unit build a city
    can_build = 1 if unit.can_build(game_state.map) else 0

    return [closest_resource_dist, space_left, closest_city_dist, closest_city_fuel, closest_city_light_upkeep, cycle_position, can_build]


def agent(observation, configuration):
    global game_state

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    
    actions = []

    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height

    resource_tiles: list[Cell] = []
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)

    # we iterate over all our units and do something with them
    for unit in player.units:
        
        # Test
        #with open("log.txt", "a") as f:
        #    print(get_input(observation, game_state, unit), file=f)

        if unit.is_worker() and unit.can_act():
            closest_dist = math.inf
            closest_resource_tile = None
            if unit.get_cargo_space_left() > 0:
                # if the unit is a worker and we have space in cargo, lets find the nearest resource tile and try to mine it
                for resource_tile in resource_tiles:
                    if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
                    if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
                    dist = resource_tile.pos.distance_to(unit.pos)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_resource_tile = resource_tile
                if closest_resource_tile is not None:
                    actions.append(unit.move(unit.pos.direction_to(closest_resource_tile.pos)))
            else:
                # if unit is a worker and there is no cargo space left, and we have cities, lets return to them
                if len(player.cities) > 0:
                    closest_dist = math.inf
                    closest_city_tile = None
                    for k, city in player.cities.items():
                        for city_tile in city.citytiles:
                            dist = city_tile.pos.distance_to(unit.pos)
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_city_tile = city_tile
                    if closest_city_tile is not None:
                        move_dir = unit.pos.direction_to(closest_city_tile.pos)
                        actions.append(unit.move(move_dir))

    # you can add debug annotations using the functions in the annotate object
    # actions.append(annotate.circle(0, 0))
    
    return actions
