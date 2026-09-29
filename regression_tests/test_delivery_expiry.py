"""Run with python -m unittest discover -s regression_tests -v.

Load the actual queue/delivery and image extraction methods without starting
AstrBot or making network requests; use the plugin's existing BeautifulSoup dependency.
"""

import ast
import asyncio
import copy
import logging
import re
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import parse_qs, quote, urlparse

from bs4 import BeautifulSoup


class MessageChain:
    def __init__(self):
        self.chain = []

    def message(self, text):
        self.chain.append(text)
        return self


def load_monitor():
    tree = ast.parse(Path(__file__).resolve().parents[1].joinpath("main.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WeiboMonitor")
    methods = {
        "_get_config", "_parse_weibo_time", "_should_skip_by_post_age",
        "_persist_discovered_posts", "_queue_pending_delivery",
        "_enqueue_pending_deliveries", "_discard_expired_deliveries",
        "_push_consumer", "_send_post_to_targets", "_collect_new_posts",
        "_resolve_mblog_text_html", "_extract_image_urls",
        "_extract_inline_image_urls", "_normalize_inline_image_url",
    }
    cls.bases = []
    cls.decorator_list = []
    cls.body = [node for node in cls.body if getattr(node, "name", "") in methods]
    constants = [node for node in tree.body if isinstance(node, ast.Assign)]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *constants, cls], type_ignores=[])
    namespace = dict(globals(), Comp=SimpleNamespace(
        Image=lambda **kw: ("image", kw),
        Video=SimpleNamespace(fromFileSystem=lambda **kw: ("video", kw)),
    ))
    exec(compile(ast.fix_missing_locations(module), "main.py", "exec"), namespace)
    return namespace["WeiboMonitor"]


WeiboMonitor = load_monitor()


class DeliveryExpiryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = datetime(2026, 9, 11, 12, tzinfo=timezone(timedelta(hours=8)))
        self.monitor = WeiboMonitor()
        m = self.monitor
        m.config = {"filter_settings": {"max_post_age_minutes": 30}}
        m._get_utc8_now = lambda: self.now
        m.plugin_logger = Mock(spec=logging.Logger)
        m.push_queue = asyncio.Queue(maxsize=100)
        m._queued_delivery_ids = set()
        m._data = {"_pending_deliveries": {}, "last_id_1": "999"}
        self.saved = None

        def save():
            self.saved = copy.deepcopy(m._data)
            return True

        m._save_data = Mock(side_effect=save)
        m._log_to_daily_file = Mock()
        m.message_format = "{weibo}"
        m._format_post_text = lambda post, fmt: post["text"]
        m._download_post_images = AsyncMock(return_value=[])
        m._download_post_video = AsyncMock(return_value=None)
        m._send_message_with_timeout = AsyncMock()
        m.context = SimpleNamespace(send_message=AsyncMock())

    def add_post(self, minutes=5, post_id="100", targets=None):
        post = {
            "_post_id": post_id, "text": "test", "username": "tester",
            "created_at": (self.now - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S"),
            "image_urls": [], "video_info": None,
        }
        delivery_id = self.monitor._persist_discovered_posts(
            "1", [post], targets or ["group1"], "{weibo}", None,
        )[0]
        return delivery_id

    async def consume(self):
        task = asyncio.create_task(self.monitor._push_consumer())
        try:
            await asyncio.wait_for(self.monitor.push_queue.join(), timeout=2)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # 队列已在恢复扫描中清空时，消费者可能尚未开始执行。
                pass

    def test_restart_removes_expired_even_with_future_retry(self):
        old = self.add_post(60, "100")
        fresh = self.add_post(5, "101")
        self.monitor._data["_pending_deliveries"][old]["next_retry_at"] = "2026-09-12 12:00:00"
        self.monitor._enqueue_pending_deliveries()
        self.assertNotIn(old, self.saved["_pending_deliveries"])
        self.assertEqual(self.saved["last_id_1"], "999")
        self.assertEqual(self.monitor._queued_delivery_ids, {fresh})
        self.monitor._log_to_daily_file.assert_not_called()

    async def test_post_expires_while_queued(self):
        delivery_id = self.add_post(29)
        self.monitor._enqueue_pending_deliveries()
        self.now += timedelta(minutes=2)
        await self.consume()
        self.monitor._send_message_with_timeout.assert_not_awaited()
        self.monitor._download_post_images.assert_not_awaited()
        self.assertEqual(self.saved["_pending_deliveries"], {})
        self.assertEqual(self.monitor._queued_delivery_ids, set())
        self.monitor._log_to_daily_file.assert_not_called()

    async def test_failed_delivery_expires_before_retry(self):
        delivery_id = self.add_post(29)
        self.monitor._send_message_with_timeout.side_effect = RuntimeError("offline")
        self.monitor._enqueue_pending_deliveries()
        await self.consume()
        self.assertEqual(self.saved["_pending_deliveries"][delivery_id]["attempts"], 1)
        self.now += timedelta(minutes=2)
        self.monitor._enqueue_pending_deliveries()
        self.assertEqual(self.saved["_pending_deliveries"], {})
        self.assertTrue(self.monitor.push_queue.empty())
        self.monitor._send_message_with_timeout.assert_awaited_once()

    async def test_expiry_during_image_download(self):
        self.add_post(29)

        async def download(post):
            self.now += timedelta(minutes=2)
            return ["image.jpg"]

        self.monitor._download_post_images.side_effect = download
        self.monitor._enqueue_pending_deliveries()
        await self.consume()
        self.monitor._send_message_with_timeout.assert_not_awaited()
        self.assertEqual(self.saved["_pending_deliveries"], {})
        self.monitor._log_to_daily_file.assert_not_called()

    async def test_expiry_between_targets(self):
        self.add_post(29, targets=["group1", "group2"])

        async def send(target, chain):
            self.now += timedelta(minutes=2)

        self.monitor._send_message_with_timeout.side_effect = send
        self.monitor._enqueue_pending_deliveries()
        await self.consume()
        self.monitor._send_message_with_timeout.assert_awaited_once()
        self.assertEqual(self.saved["_pending_deliveries"], {})
        self.assertEqual(self.monitor._log_to_daily_file.call_args.kwargs["delivery_count"], 1)

    async def test_expiry_during_video_download(self):
        delivery_id = self.add_post(29)
        self.monitor._data["_pending_deliveries"][delivery_id]["post"]["video_info"] = {"url": "video"}

        async def download(post):
            self.now += timedelta(minutes=2)
            return "video.mp4"

        self.monitor._download_post_video.side_effect = download
        self.monitor._enqueue_pending_deliveries()
        await self.consume()
        self.monitor._send_message_with_timeout.assert_awaited_once()
        self.monitor.context.send_message.assert_not_awaited()
        self.assertEqual(self.saved["_pending_deliveries"], {})

    async def test_cleanup_save_failure_still_does_not_send(self):
        delivery_id = self.add_post(60)
        self.monitor._save_data.side_effect = None
        self.monitor._save_data.return_value = False
        self.monitor._queue_pending_delivery(delivery_id)
        self.monitor._enqueue_pending_deliveries()
        await self.consume()
        self.assertIn(delivery_id, self.monitor._data["_pending_deliveries"])
        self.monitor._send_message_with_timeout.assert_not_awaited()

    async def test_disabled_filter_keeps_old_delivery(self):
        self.add_post(120)
        self.monitor.config["filter_settings"]["max_post_age_minutes"] = 0
        self.monitor._enqueue_pending_deliveries()
        await self.consume()
        self.monitor._send_message_with_timeout.assert_awaited_once()
        self.assertEqual(self.saved["_pending_deliveries"], {})

    async def test_legacy_timestamp_is_filtered_on_restart_and_in_queue(self):
        for raw in [
            "2026-09-10 10:00:00 00:00:00",
            "2026-09-10T02:00:00Z 00:00:00",
            "2026-09-10 10:00 00:00:00",
        ]:
            for already_queued in [False, True]:
                with self.subTest(raw=raw, already_queued=already_queued):
                    self.setUp()
                    m = self.monitor
                    delivery_id = self.add_post(120)
                    m._data["_pending_deliveries"][delivery_id]["post"]["created_at"] = raw
                    if already_queued:
                        m._queue_pending_delivery(delivery_id)
                    else:
                        m._enqueue_pending_deliveries()
                    await self.consume()
                    m._send_message_with_timeout.assert_not_awaited()
                    m._download_post_images.assert_not_awaited()
                    m._log_to_daily_file.assert_not_called()
                    self.assertEqual(self.saved["_pending_deliveries"], {})
                    self.assertEqual(self.saved["last_id_1"], "999")

    async def test_fresh_legacy_timestamp_is_not_discarded(self):
        delivery_id = self.add_post(5)
        self.monitor._data["_pending_deliveries"][delivery_id]["post"]["created_at"] = (
            "2026-09-11T03:55:00Z 00:00:00"
        )
        self.monitor._enqueue_pending_deliveries()
        await self.consume()
        self.monitor._send_message_with_timeout.assert_awaited_once()

    def test_invalid_legacy_timestamp_is_still_reported(self):
        self.assertFalse(self.monitor._should_skip_by_post_age(
            "2026-99-99 10:00:00 00:00:00", 100
        ))
        self.monitor.plugin_logger.warning.assert_called_once()

    async def test_cleanup_failure_during_send_preserves_delivery_confirmation(self):
        for partial_failure in [False, True]:
            with self.subTest(partial_failure=partial_failure):
                self.setUp()
                m = self.monitor
                fresh = self.add_post(5, "101", ["group1", "group2"])
                old = self.add_post(120, "100")
                m._queue_pending_delivery(fresh)
                original_save = m._save_data.side_effect
                data = m._data
                pending = data["_pending_deliveries"]
                old_item = pending[old]

                async def send(target, chain):
                    if target == "group1":
                        # 模拟正文发送等待期间，监控循环清理其他过期待办失败。
                        m._save_data.side_effect = lambda: False
                        try:
                            m._enqueue_pending_deliveries()
                        finally:
                            m._save_data.side_effect = original_save
                    elif partial_failure:
                        raise RuntimeError("offline")

                m._send_message_with_timeout.side_effect = send
                await self.consume()
                self.assertIs(m._data, data)
                self.assertIs(m._data["_pending_deliveries"], pending)
                self.assertIs(pending[old], old_item)
                if partial_failure:
                    item = self.saved["_pending_deliveries"][fresh]
                    self.assertEqual(item["delivered_targets"], ["group1"])
                    self.assertEqual(item["pending_targets"], ["group2"])
                else:
                    self.assertNotIn(fresh, self.saved["_pending_deliveries"])

                self.now += timedelta(minutes=2)
                m._send_message_with_timeout.side_effect = None
                m._enqueue_pending_deliveries()
                await self.consume()
                sent_targets = [call.args[0] for call in m._send_message_with_timeout.await_args_list]
                self.assertEqual(sent_targets, ["group1", "group2"] + (["group2"] if partial_failure else []))
                self.assertEqual(self.saved["_pending_deliveries"], {})
                m._log_to_daily_file.assert_called_once()

    async def test_recheck_current_setting_after_enqueue(self):
        self.monitor.config["filter_settings"]["max_post_age_minutes"] = 0
        self.add_post(120)
        self.monitor._enqueue_pending_deliveries()
        self.monitor.config["filter_settings"]["max_post_age_minutes"] = 30
        await self.consume()
        self.monitor._send_message_with_timeout.assert_not_awaited()
        self.assertEqual(self.saved["_pending_deliveries"], {})

    def test_time_formats_and_boundary(self):
        for raw in ["2026-09-11 11:30:00", "2026-09-11T03:30:00Z", "Fri Sep 11 03:30:00 +0000 2026"]:
            with self.subTest(raw=raw):
                parsed = self.monitor._parse_weibo_time(raw)
                self.assertEqual(parsed, "2026-09-11 11:30:00")
                self.assertFalse(self.monitor._should_skip_by_post_age(parsed, 100))
        self.assertTrue(self.monitor._should_skip_by_post_age("2026-09-11 11:29:59", 100))
        self.assertFalse(self.monitor._should_skip_by_post_age("unknown", 100))

    async def test_new_post_filter_accepts_normalized_dates(self):
        m = self.monitor
        m._resolve_mblog_text_html = AsyncMock(return_value="test")
        m.clean_text = lambda text: text
        m._has_filter_keyword = lambda *args: False
        m._should_skip_by_whitelist = lambda *args: False
        m._extract_video_info = lambda post: None
        posts = [
            {"id": "102", "bid": "fresh", "created_at": "2026-09-11 11:59:00"},
            {"id": "101", "bid": "old", "created_at": "2026-09-11 10:00:00"},
        ]
        collected = await m._collect_new_posts("1", posts, 100, False, "tester")
        self.assertEqual([p["_post_id"] for p in collected], ["102"])

    async def test_full_text_only_image_is_collected_when_enabled(self):
        image_url = "https://wx1.sinaimg.cn/large/example.jpg"
        full_html = f'完整正文 <a href="{image_url}">查看图片</a>'
        for enabled in [True, False]:
            with self.subTest(enabled=enabled):
                self.setUp()
                m = self.monitor
                m.config["content_settings"] = {"show_full_weibo_text": enabled}
                m._request_semaphore = asyncio.Semaphore(1)
                m.get_headers = lambda uid: {}
                response = SimpleNamespace(
                    status_code=200,
                    json=lambda: {"ok": 1, "data": {"longTextContent": full_html}},
                )
                m.client = SimpleNamespace(get=AsyncMock(return_value=response))
                m.clean_text = lambda html: BeautifulSoup(html, "html.parser").get_text()
                m._has_filter_keyword = lambda *args: False
                m._should_skip_by_whitelist = lambda *args: False
                m._extract_video_info = lambda post: None
                post = {
                    "id": "102", "bid": "fresh", "text": "摘要", "pics": [],
                    "isLongText": True, "created_at": "2026-09-11 11:59:00",
                }

                collected = await m._collect_new_posts("1", [post], 100, False, "tester")

                self.assertEqual(len(collected), 1)
                self.assertEqual(collected[0]["text"], "完整正文 查看图片" if enabled else "摘要")
                self.assertEqual(collected[0]["image_urls"], [image_url] if enabled else [])
                if enabled:
                    m.client.get.assert_awaited_once()
                else:
                    m.client.get.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
