"""Sentinel Edge — on-premises AI video-analytics pipeline.

Raw video never leaves the site: pixels are decoded, analysed and discarded
here; only event metadata + a 10s clip + a thumbnail go to the cloud, over
outbound-only connections.
"""

__version__ = "0.1.0"
