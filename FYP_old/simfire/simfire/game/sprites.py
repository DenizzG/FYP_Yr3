import os
import sys
import tempfile
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pygame
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg
from wurlitzer import pipes
import matplotlib.pyplot as plt
from skopt import gp_minimize


from ..enums import BURNED_RGB_COLOR, BurnStatus, SpriteLayer
from ..utils.layers import FuelLayer, HistoricalLayer, TopographyLayer
from ..utils.log import create_logger
from math import gcd


log = create_logger(__name__)


class Terrain(pygame.sprite.Sprite):
    """
    Use a TopographyLayer and a FuelLayer to make terrain. This sprite is just the
    entire background that the fire appears on. This sprite changes the color of each
    terrain pixel based on its "dryness/flammability" if it is unburned, or based on
    whether the pixel is a fireline or burned.
    """

    def __init__(
        self,
        fuel_layer: FuelLayer,
        topo_layer: TopographyLayer,
        screen_size: Tuple[int, int],
        headless: bool = False,
        historical_layer: Optional[HistoricalLayer] = None,
    ) -> None:

        super().__init__()

        self.fuel_layer = fuel_layer
        self.topo_layer = topo_layer
        self.historical_layer = historical_layer

        self.screen_size = screen_size
        self.headless = headless

        self.elevations = self.topo_layer.data.squeeze()
        self.fuels = self.fuel_layer.data.squeeze()

        self.image: Optional[pygame.surface.Surface]
        self.rect: Optional[pygame.Rect]

        self.rect = pygame.Rect(0, 0, *self.screen_size)
        if not self.headless:
            # Create the terrain image
            terrain_image = self._make_terrain_image()
            # Convert the terrain image to a PyGame surface for display
            self.image = pygame.surfarray.make_surface(terrain_image.swapaxes(0, 1))
            # The rectangle for this sprite is the entire game
        else:
            self.image = None

        # This sprite should always have layer 1 since it will always
        # be behind every other sprite
        self.layer = SpriteLayer.TERRAIN

    def copy(self):
        """
        Creates a new Terrain object with the same data/attributes.
        Recreates the PyGame surface so it can render properly.
        """
        new_terrain = Terrain(
            self.fuel_layer,
            self.topo_layer,
            self.screen_size,
            headless=False,  # Allow rendering
            historical_layer=self.historical_layer,
        )
        # The following attributes are created in the parent constructor, but
        # we want to make sure they are the same as the original
        new_terrain.elevations = self.elevations.copy()
        new_terrain.fuels = self.fuels.copy()

        return new_terrain

    def update(self, *args: Any, **kwargs: Any) -> None:
        """
        Change any burned squares to brown using fire_map, which
        contains pixel-wise values for tile burn status.
        The parent class update() expects just args and kwargs, so we

        Arguments:
            a
            fire_map: A 2-D numpy array containing enumerated values for unburned,
                      burning, and burned tile status for the game screen

        Returns: None
        """
        # Argument checks for compatibility
        if len(args) == 0 or len(args) > 1:
            raise ValueError(
                "The input arguments to update() should contain"
                "only one value for the fire_map. Instead got: "
                f"{args}"
            )
        if len(kwargs) > 0:
            raise ValueError(
                "The input keyword arguments to update() should "
                f"contain no values. Instead got: {kwargs}"
            )
        if not isinstance(args[0], np.ndarray):
            raise TypeError(
                "The input fire_map should be a numpy array. " f"Instead got: {args[0]}"
            )

        fire_map: np.ndarray = args[0]
        if fire_map.shape != self.screen_size:
            raise ValueError(
                f"The shape of the fire_map {fire_map.shape} does "
                f"not match the shape of the screen {self.screen_size}"
            )
        fire_map = fire_map.copy()
        self._update(fire_map)

    def _update(self, fire_map: np.ndarray) -> None:
        """
        Internal method to update the burned squares to brown. This will update
        self.image in-place.
        This is needed because the parent class Sprite.update() only take in
        args and kwargs, and we want to do input checking on those parameters
        before updating.

        Arguments:
            fire_map: A 2-D numpy array containing enumerated values for unburned,
                      burning, and burned tile status for the game screen
        """
        burned_idxs = np.where(fire_map == BurnStatus.BURNED)
        if not self.headless:
            # This method will update self.image in-place with arr
            if self.image is not None:
                arr = pygame.surfarray.pixels3d(self.image)
            arr[burned_idxs[::-1]] = BURNED_RGB_COLOR

    def _make_terrain_image(self) -> np.ndarray:
        """
        Use the FuelLayer image and TopographyLayer contours to create the
        terrain background image. This will show the FuelLayer as the landscape/overhead
        view, with the contour lines overlaid on top.

        Arguments:
            None

        Returns:
            out_image: The input image with the contour lines drawn on it
        """
        image = self.fuel_layer.image.squeeze()

        # Create a figure with axes
        fig, ax = plt.subplots()

        # the decimal points look messy)
        contours = ax.contour(
            self.topo_layer.data.squeeze(), origin="upper", colors="black"
        )

        # The fmt argument will display the levels as whole numbers (otherwise
        # the decimal points look messy)
        ax.clabel(
            contours,
            contours.levels,
            inline=True,
            fmt=lambda x: f"{x:.0f}",
            fontsize="small",
        )
        ax.imshow(image.astype(np.uint8), alpha=np.zeros(image.shape[:2]))

        # plot the historical perimeters
        # TODO: Not plotting in the correct order
        # if self.historical_layer:
        #     # now we can make the perimater image
        #     perimeter_image = self.historical_layer._make_perimeters_image()
        #     ax.imshow(perimeter_image[...,:3].astype(np.uint8),
        #               alpha=np.uint8(perimeter_image[...,3]/255))

        plt.axis("off")

        # Save the figure as a vector graphic to get just the image (no axes,
        # ticks, figure edges, etc.)
        # Then load it, resize, and convert to numpy
        # Added `with pipes():` to get rid of 'colinear!' message output by C library
        # in svglib:
        # https://github.com/Distrotech/reportlab/search?q=colinear%21
        with pipes():
            with tempfile.NamedTemporaryFile(suffix=".svg") as out_img_path:
                fig.savefig(out_img_path.name, bbox_inches="tight", pad_inches=0)

                drawing = svg2rlg(out_img_path.name)

                # Resize the SVG drawing
                scale_x = image.shape[1] / drawing.width
                scale_y = image.shape[0] / drawing.height
                drawing.width = drawing.width * scale_x
                drawing.height = drawing.height * scale_y
                drawing.scale(scale_x, scale_y)
                # Convert to Pillow
                # The fmt argument will display the levels as whole numbers (otherwise
                # sending stdout to devnull to avoid the annoying `x_order_1: collinear!`
                sys.stdout = open(os.devnull, "w")
                out_img_pil = renderPM.drawToPIL(drawing)
                sys.stdout = sys.__stdout__
        plt.close(fig)
        # Slice the alpha channel off
        out_img = np.array(out_img_pil, dtype=np.uint8)[..., :3]

        return out_img


class Fire(pygame.sprite.Sprite):
    """
    This sprite represents a fire burning on one pixel of the terrain. Its
    image is generally kept very small to make rendering easier. All fire
    spreading is handled by the FireManager it is attached to.
    """

    def __init__(self, pos: Tuple[int, int], size: int, headless: bool = False) -> None:
        """
        Initialize the class by recording the position and size of the sprite
        and creating a solid color texture.

        Arguments:
            pos: The (x, y) pixel position of the sprite
            size: The pixel size of the sprite
            headless: Flag to run in a headless state. This will allow PyGame objects to
                      not be initialized.
        """
        super().__init__()

        self.pos = pos
        self.size = size
        self.headless = headless
        self.rect: pygame.rect.Rect

        if self.headless:
            self.image = None
            # Need to use self.rect to track the location of the sprite
            # When running headless, we need this to be a tuple instead of a PyGame Rect
            self.rect = pygame.Rect(*(pos + (size, size)))
        else:
            fire_color = np.zeros((self.size, self.size, 3))
            fire_color[:, :, 0] = 255
            fire_color[:, :, 1] = 153
            fire_color[:, :, 2] = 51
            self.image = pygame.surfarray.make_surface(fire_color)

            self.rect = self.image.get_rect()
            self.rect = self.rect.move(self.pos[0], self.pos[1])

        # Layer 3 so that it appears on top of the terrain and line (if applicable)
        self.layer: int = SpriteLayer.FIRE

    def update(self, *args, **kwargs) -> None:
        """
        Currently unused.
        """
        pass


class FireLine(pygame.sprite.Sprite):
    """
    This sprite represents a fireline on one pixel of the terrain. Its image is generally
    kept very small to make rendering easier. All fireline placement spreading is handled
    by the FireLineManager it is attached to.
    """

    def __init__(self, pos: Tuple[int, int], size: int, headless: bool = False) -> None:
        """
        Initialize the class by recording the position and size of the sprite
        and creating a solid color texture.

        Arguments:
            pos: The (x, y) pixel position of the sprite
            size: The pixel size of the sprite
            headless: Flag to run in a headless state. This will allow PyGame objects to
                      not be initialized.

        """
        super().__init__()

        self.pos = pos
        self.size = size
        self.headless = headless

        if self.headless:
            self.image = None
            # Need to use self.rect to track the location of the sprite
            # When running headless, we need this to be a tuple instead of a PyGame Rect
            self.rect = pygame.Rect(*(pos + (size, size)))
        else:
            fireline_color = np.zeros((self.size, self.size, 3))
            fireline_color[:, :, 0] = 255  # R
            fireline_color[:, :, 1] = 0  # G
            fireline_color[:, :, 2] = 0  # B
            self.image = pygame.surfarray.make_surface(fireline_color)

            self.rect = self.image.get_rect()
            self.rect = self.rect.move(self.pos[0], self.pos[1])

        # Layer LINE so that it appears on top of the terrain
        self.layer: int = SpriteLayer.LINE

    def update(self, *args, **kwargs) -> None:
        """
        This doesn't require to be updated right now. May change in the future if we
        learn new things about the physics.
        """
        pass


class ScratchLine(pygame.sprite.Sprite):
    """
    This sprite represents a scratch line on one pixel of the terrain. Its image is
    generally kept very small to make rendering easier. All scratch line placement
    spreading is handled by the ScratchLineManager it is attached to.
    """

    def __init__(self, pos: Tuple[int, int], size: int, headless: bool = False) -> None:
        """
        Initialize the class by recording the position and size of the sprite
        and creating a solid color texture.
        """
        super().__init__()

        self.pos = pos
        self.size = size
        self.headless = headless

        if self.headless:
            self.image = None
            # Need to use self.rect to track the location of the sprite
            # When running headless, we need this to be a tuple instead of a PyGame Rect
            self.rect = pygame.Rect(*(pos + (size, size)))
        else:
            scratchline_color = np.zeros((self.size, self.size, 3))
            scratchline_color[:, :, 0] = 255  # R
            scratchline_color[:, :, 1] = 0  # G
            scratchline_color[:, :, 2] = 0  # B
            self.image = pygame.surfarray.make_surface(scratchline_color)

            self.rect = self.image.get_rect()
            self.rect = self.rect.move(self.pos[0], self.pos[1])

        # Layer LINE so that it appears on top of the terrain
        self.layer: int = SpriteLayer.LINE

    def update(self, *args, **kwargs) -> None:
        """
        This doesn't require to be updated right now. May change in the future if we
        learn new things about the physics.

        """
        pass


class WetLine(pygame.sprite.Sprite):
    """
    This sprite represents a wet line on one pixel of the terrain. Its image is
    generally kept very small to make rendering easier. All wet line placement
    spreading is handled by the WaterLineManager it is attached to.
    """

    def __init__(self, pos: Tuple[int, int], size: int, headless: bool = False) -> None:
        """
        Initialize the class by recording the position and size of the sprite
        and creating a color texture.
        """
        super().__init__()

        self.pos = pos
        self.size = size
        self.headless = headless

        if self.headless:
            self.image = None
            # Need to use self.rect to track the location of the sprite
            # When running headless, we need this to be a tuple instead of a PyGame Rect
            self.rect = pygame.Rect(*(pos + (size, size)))
        else:
            wetline_color = np.zeros((self.size, self.size, 3))
            wetline_color[:, :, 0] = 212  # R
            wetline_color[:, :, 1] = 241  # G
            wetline_color[:, :, 2] = 249  # B
            self.image = pygame.surfarray.make_surface(wetline_color)

            self.rect = self.image.get_rect()
            self.rect = self.rect.move(self.pos[0], self.pos[1])

        # Layer LINE so that it appears on top of the terrain
        self.layer: int = SpriteLayer.LINE

    def update(self, *args, **kwargs) -> None:
        """
        This doesn't require to be updated right now. May change in the future if we
        learn new things about the physics.

        """
        pass


class Agent(pygame.sprite.Sprite):
    """
    This sprite represents an agent (e.g. firefighter, mitigation unit) on one pixel of the terrain.
    Its image is generally kept very small to make rendering easier.
    All agent movement and logic are handled by the simulation or an external controller.

    agent_id : int
        The unique ID of this agent.
   
    initial_position : tuple[int, int]
        The (x,y) starting position of the agent, where (0,0) is the top-left corner of
        the map and (max_x, max_y) is the bottom-right corner of the map.

    latest_movement : str or None
        The last movement made by the agent, if applicable.
    latest_interaction : str or None
        The last interaction had by the agent, if applicable.
    mitigation_placed : bool
        Whether the agent has placed any mitigations recently.
    moved_off_map : bool
        Whether the agent has moved off the map recently.

    """

    def __init__(
        self, 
        pos: Tuple[int, int], 
        size: int, 
        agent_id: str, 
        #sim_id: int, 
        fire_map_shape: Tuple[int, int],
        headless: bool = False, 
        ) -> None:

        """
        Initialize the class by recording the position and size of the sprite
        and creating a solid color texture.

        Arguments:
            pos: The (x, y) pixel position of the sprite
            size: The pixel size of the sprite
            headless: Flag to run in a headless state. This will allow PyGame objects to
                      not be initialized.
        """
        super().__init__()

        #sim_id tells you which simulation instance the agent belongs to
        #self.sim_id = sim_id #contained inside sim.agents.keys(), 
        self.agent_id = agent_id
        self._pos = pos
        self.agent_color: Optional[np.ndarray] = None
        self.size = size
        self.headless = headless
        self.fire_map_shape = fire_map_shape  # <-- Add this line


        self._previous_position: Tuple[int,int] = None 
        self.latest_movement: str = "down"
        self.latest_interaction: str = "fireline"
        self.mitigation_placed: bool = False
        self.moved_off_map: bool = False
        self.action_space = []
        #self.actions = []
        self.current_action = 0
        self.waypoints: Optional[list[Tuple[int, int]]] = None
        self.touched_fire: int = 0
        self._move_counter = {"x": 0, "y": 0}
        self.valid_points = []

        #rect.x, rect.y         # Position (top-left corner)
        #rect.width, rect.height  # Size
        #rect.center            # Center point
        #rect.colliderect(other_rect)  # Check if it overlaps with another
           
        self.rect: pygame.rect.Rect 

        # Create array used to store coords adjacent to "true" mitigations placed by.
        self.adj_to_mitigation = np.zeros(self.fire_map_shape, dtype=bool)


        if self.headless: #if not running on a server (always for this application)
            self.image = None
            # Need to use self.rect to track the location of the sprite
            # When running headless, we need this to be a tuple instead of a PyGame Rect
            self.rect = pygame.Rect(*(pos + (size, size)))
        else:
            if self.agent_color is None:
                self.agent_color = np.zeros((self.size, self.size, 3))
                self.agent_color[:, :, 0] = 0
                self.agent_color[:, :, 1] = 0
                self.agent_color[:, :, 2] = 255
            self.image = pygame.surfarray.make_surface(self.agent_color)
            self.rect = self.image.get_rect()
            self.rect.update(*(self.pos + (size, size)))

        # Layer 3 so that it appears on top of the terrain and line (if applicable)
        self.layer: int = SpriteLayer.AGENT

    @property
    def pos(self) -> Tuple[int, int]:
        return self._pos

    @property
    def previous_position(self) -> Tuple[int, int]:
        return self._previous_position

    @pos.setter
    def pos(self, value: Tuple[int, int]) -> None:
        self._previous_position = self.pos
        self._pos = value
        self.rect.update(*(self.pos + (self.size, self.size)))

    @property
    def x(self) -> int:
        return self._pos[0]

    @property
    def y(self) -> int:
        return self._pos[1]

    def reset(self):
        self.latest_movement = None
        self.latest_interaction = None
        self.mitigation_placed = False
        self.__init__()

    def actions(self, action: str, interaction: str) -> None:
        """
        Store the action to be executed later during the simulation update.

        Args:
            action (str): One of 'up', 'down', 'left', 'right', 'fireline', 'wetline', or 'scratchline'.
        """
        valid_actions = {"up", "down", "left", "right"}
        valid_interactions = {"fireline", "wetline", "scratchline"}
        if action in valid_actions:
            self.latest_movement = action
        elif action == None:
            pass
        else:
            log.warning(f"Invalid action '{action}' for agent {self.agent_id}")
        
        if interaction in valid_interactions:
            self.latest_interaction = interaction
        elif interaction == None:
            pass
        else:
            log.warning(f"Invalid interaction '{interaction}' for agent {self.agent_id}")
            
    def next_movement(self) -> None:
        """
        Move the agent towards the current waypoint in self.waypoints.
        The agent moves in cardinal directions, choosing the direction that
        best approximates a straight line to the waypoint using the ratio of dx:dy.
        For every N steps in x, take M steps in y, where N:M approximates |dx|:|dy|.
        Handles all quadrants (positive and negative directions).
        """
        if not hasattr(self, "waypoints") or not self.waypoints:
            self.latest_movement = None
            return

        target = self.waypoints[0]
        x, y = self.pos
        tx, ty = target

        dx = tx - x
        dy = ty - y

        if dx == 0 and dy == 0:
            # Waypoint reached, pop and check next
            self.waypoints.pop(0)
            self.latest_movement = None
            if hasattr(self, "_move_counter"):
                self._move_counter = {"x": 0, "y": 0}
            return

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        # If either dx or dy is zero, just move in the nonzero direction
        if abs_dx == 0:
            self.latest_movement = "down" if dy > 0 else "up"
            return
        if abs_dy == 0:
            self.latest_movement = "right" if dx > 0 else "left"
            return

        # Find the smallest integer ratio that approximates dx:dy
        g = gcd(abs_dx, abs_dy)
        step_x = abs_dx // g
        step_y = abs_dy // g

        # Store direction for x and y
        dir_x = "right" if dx > 0 else "left"
        dir_y = "down" if dy > 0 else "up"

        # Use counters to track how many steps have been taken in each direction
        if not hasattr(self, "_move_counter"):
            self._move_counter = {"x": 0, "y": 0}

        # Decide which direction to move next based on the ratio
        # The idea: for every step_x steps in x, take step_y steps in y
        # We alternate between x and y moves to approximate the line

        # Calculate the total steps needed to reach the next "corner" in the grid
        total_steps = step_x + step_y

        # Determine which direction to move this step
        # Use the counters to keep track of progress along the ratio
        if self._move_counter["x"] * step_y <= self._move_counter["y"] * step_x:
            # Move in x direction (respect sign)
            self.latest_movement = dir_x
            self._move_counter["x"] += 1
        else:
            # Move in y direction (respect sign)
            self.latest_movement = dir_y
            self._move_counter["y"] += 1

        # Reset counters if we've completed a full ratio cycle or reached the waypoint
        if (self._move_counter["x"] >= step_x and self._move_counter["y"] >= step_y) or \
           (abs(self.pos[0] - tx) < step_x and abs(self.pos[1] - ty) < step_y):
            self._move_counter = {"x": 0, "y": 0}

        

    def update(self, *args, **kwargs) -> None:
        """
        Update the agent's state based on the action.
        Valid actions include movement and mitigation placement.
        """

        x, y = self.pos

        if self.latest_movement == "up":
            y = max(0, y - 1)
        elif self.latest_movement == "down":
            y = min(self.fire_map_shape[0] - 1, y + 1)
        elif self.latest_movement == "left":
            x = max(0, x - 1)
        elif self.latest_movement == "right":
            x = min(self.fire_map_shape[1] - 1, x + 1)

        if self.latest_interaction == "fireline":
            self.mitigation_placed = True
        elif self.latest_interaction == "wetline":
            self.mitigation_placed = True
        elif self.latest_interaction == "scratchline":
            self.mitigation_placed = True

        # Update the agent's position
        self.pos = (x, y)

    def copy_agent(self):
        """
        Custom copy for Agent that avoids pygame surfaces and sprite group leakage.
        """
        agent_copy = Agent(
            pos=self.pos,
            size=self.size,
            agent_id=self.agent_id,
            fire_map_shape=self.fire_map_shape,
            headless=self.headless
        )
        
        # Manually copy simple fields
        agent_copy.action_space = list(self.action_space)
        agent_copy.touched_fire = self.touched_fire
        agent_copy.latest_interaction = self.latest_interaction
        agent_copy.valid_points = list(self.valid_points)
        agent_copy.waypoints = None
        agent_copy._move_counter = dict(self._move_counter)
        agent_copy._previous_position = self._previous_position
        agent_copy.mitigation_placed = self.mitigation_placed
        agent_copy.moved_off_map = self.moved_off_map

        return agent_copy


