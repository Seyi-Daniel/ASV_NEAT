import pygame
import asyncio
from game_manager import GameManager

# This is the single, combined main entry point
async def main():
    game = GameManager(1000, 1000) # Width/Height
    await game.run_async()

if __name__ == "__main__":
    asyncio.run(main())