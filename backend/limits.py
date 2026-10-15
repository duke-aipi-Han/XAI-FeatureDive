"""Bound API request bodies and process-local request frequency."""
from collections import OrderedDict, deque
import json
from math import ceil
import threading
import time


class RequestLimits:
    def __init__(self, app, max_body_bytes=131072, requests_per_minute=120, max_rate_clients=1024):
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.requests_per_minute = requests_per_minute
        self.max_rate_clients = max_rate_clients
        self.clients = OrderedDict()
        self.lock = threading.Lock()

    async def reject(self, send, status, detail, retry_after=None):
        headers = [(b'content-type', b'application/json')]
        if retry_after is not None:
            headers.append((b'retry-after', str(retry_after).encode()))
        await send({'type': 'http.response.start', 'status': status, 'headers': headers})
        await send({'type': 'http.response.body', 'body': json.dumps({'detail': detail}).encode()})

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or not scope['path'].startswith('/api/') or scope['path'] in ('/api/health', '/api/ready'):
            return await self.app(scope, receive, send)
        # Trust the socket peer, never an unverified forwarded-IP header.
        peer = (scope.get('client') or ('unknown', 0))[0]
        now = time.monotonic()
        retry_after = None
        with self.lock:
            # Bound memory without evicting an active client's limit window.
            for client in list(self.clients):
                if not self.clients[client] or self.clients[client][-1] <= now - 60:
                    del self.clients[client]
                else:
                    break
            if peer not in self.clients and len(self.clients) >= self.max_rate_clients:
                retry_after = 60
            else:
                window = self.clients.setdefault(peer, deque())
                while window and window[0] <= now - 60:
                    window.popleft()
                if len(window) >= self.requests_per_minute:
                    retry_after = max(1, ceil(60 - (now - window[0])))
                else:
                    window.append(now)
                    self.clients.move_to_end(peer)
        if retry_after is not None:
            return await self.reject(send, 429, 'Request limit reached; retry later', retry_after)
        length = dict(scope.get('headers', [])).get(b'content-length')
        if length is not None:
            try:
                if int(length) < 0:
                    raise ValueError
                if int(length) > self.max_body_bytes:
                    return await self.reject(send, 413, 'Request body is too large')
            except ValueError:
                return await self.reject(send, 400, 'Invalid content length')
        body = bytearray()
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            chunk = message.get('body', b'')
            if len(body) + len(chunk) > self.max_body_bytes:
                return await self.reject(send, 413, 'Request body is too large')
            body.extend(chunk)
            if not message.get('more_body', False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            return await receive()

        await self.app(scope, bounded_receive, send)
