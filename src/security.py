"""SSRF prevention and URL safety checks."""

import ipaddress
import socket
from urllib.parse import urlparse

from src.errors import DownloadError

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("ff00::/8"),
]

_BLOCKED_HOSTNAMES = {
    "metadata.google.internal",
    "metadata.goog",
}


def validate_url(url: str) -> str:
    """Validate a URL for SSRF safety. Returns the URL if safe, raises otherwise."""
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise DownloadError(
            f"Unsupported URL scheme: {parsed.scheme}",
            hint="Only http and https URLs are allowed.",
        )

    hostname = parsed.hostname
    if not hostname:
        raise DownloadError("URL has no hostname.", hint="Provide a valid URL.")

    if hostname.lower() in _BLOCKED_HOSTNAMES:
        raise DownloadError("Blocked hostname.", hint="This hostname is not allowed.")

    try:
        addrinfos = socket.getaddrinfo(hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise DownloadError(
            f"Cannot resolve hostname: {hostname}",
            hint="Check that the URL is correct and the host exists.",
        )

    for family, _type, _proto, _canonname, sockaddr in addrinfos:
        ip = ipaddress.ip_address(sockaddr[0])
        for network in _BLOCKED_NETWORKS:
            if ip in network:
                raise DownloadError(
                    "URL resolves to a blocked IP range.",
                    hint="Private/internal IPs are not allowed.",
                )

    return url
