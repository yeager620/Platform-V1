# gamelogs/feed.py

import asyncio
import datetime
from typing import List, Dict, Any, Optional, Callable
import signal
import json

from MsgCore.BaseballSavant.bs_live_gamelogs import LiveGamelogsFetcher


class LiveGamelogsFeed:
    def __init__(
            self,
            days_ahead: int = 7,
            max_concurrent_requests: int = 10,
            update_interval: int = 300,  # seconds
            callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None
    ):
        """
        Initialize the LiveGamelogsFeed.

        :param days_ahead: Number of days ahead to look for upcoming games.
        :param max_concurrent_requests: Maximum number of concurrent HTTP requests.
        :param update_interval: Time interval between updates in seconds.
        :param callback: Optional function to call with updated gamelogs.
        """
        self.fetcher = LiveGamelogsFetcher(
            days_ahead=days_ahead,
            max_concurrent_requests=max_concurrent_requests
        )
        self.update_interval = update_interval
        self.callback = callback

        # Internal storage for gamelogs
        self.gamelogs: Dict[int, Dict[str, Any]] = {}  # game_pk -> gamelog

        # Event loop and tasks
        self.loop = asyncio.get_event_loop()
        self.main_task: Optional[asyncio.Task] = None
        self.stop_event = asyncio.Event()

    async def _main_loop(self):
        """
        Main loop that continuously updates gamelogs at specified intervals.
        """
        while not self.stop_event.is_set():
            print(f"[{datetime.datetime.now()}] Fetching latest gamelogs...")
            await self._fetch_and_update()
            print(f"[{datetime.datetime.now()}] Update complete. Next update in {self.update_interval} seconds.")
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=self.update_interval)
            except asyncio.TimeoutError:
                continue  # Timeout occurred, loop continues

    async def _fetch_and_update(self):
        """
        Fetches the latest game logs and updates the internal state.
        """
        try:
            new_gamelogs = await self.fetcher.get_gamelogs_for_next_games()
            for gamelog in new_gamelogs:
                game_pk = gamelog.get('gamePk')
                if game_pk:
                    self.gamelogs[game_pk] = gamelog  # Update or add the gamelog

            # Optional callback with the latest gamelogs
            if self.callback:
                self.callback(list(self.gamelogs.values()))

        except Exception as e:
            print(f"Error during fetch and update: {e}")

    def start_feed(self):
        """
        Starts the live gamelogs feed.
        """
        print("Starting LiveGamelogsFeed...")
        self.main_task = self.loop.create_task(self._main_loop())

        # Handle graceful shutdown on SIGINT and SIGTERM
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self.loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop_feed()))
            except NotImplementedError:
                # Signal handlers are not available on some platforms (e.g., Windows)
                pass

        try:
            self.loop.run_until_complete(self.main_task)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()
            print("LiveGamelogsFeed stopped.")

    async def stop_feed(self):
        """
        Signals the main loop to stop and waits for it to finish.
        """
        print("Stopping LiveGamelogsFeed...")
        self.stop_event.set()
        if self.main_task:
            await self.main_task

    def get_current_gamelogs(self) -> List[Dict[str, Any]]:
        """
        Retrieves the current gamelogs.

        :return: List of current gamelog data.
        """
        return list(self.gamelogs.values())
