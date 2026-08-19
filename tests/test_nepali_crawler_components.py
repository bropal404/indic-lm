from crawler.deduplicate import content_hash
from crawler.frontier import is_candidate_url, normalize_url
from crawler.language import calculate_language_stats, is_predominantly_nepali
from crawler.normalize import clean_text, repair_mojibake
from crawler.parser import parse_sitemap


def test_normalize_url_removes_tracking_and_amp():
    url = "HTTPS://Example.COM/news/123/amp/?utm_source=x&keep=1#frag"
    assert normalize_url(url) == "https://example.com/news/123?keep=1"


def test_candidate_url_blocks_assets_and_search():
    assert is_candidate_url("https://example.com/news/story-1")
    assert not is_candidate_url("https://example.com/search?q=x")
    assert not is_candidate_url("https://example.com/logo.png")


def test_language_stats_for_nepali_text():
    text = "नेपालमा आज नयाँ समाचार प्रकाशित भएको छ। Kathmandu नाम पनि हुन सक्छ।"
    stats = calculate_language_stats(text)
    assert stats.devanagari_ratio > 0.6
    assert stats.latin_ratio < 0.2
    assert is_predominantly_nepali(text)


def test_clean_text_preserves_danda_and_paragraphs():
    text = "नेपाल&nbsp;&nbsp;सुन्दर छ।।।।\n\n\nयो दोस्रो अनुच्छेद हो।"
    cleaned = clean_text(text)
    assert "नेपाल सुन्दर छ" in cleaned
    assert "।।" in cleaned
    assert "\n\n" in cleaned


def test_sitemap_index_and_urlset_parse():
    urls, nested = parse_sitemap(
        b'<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><sitemap><loc>https://x.test/s.xml</loc></sitemap></sitemapindex>'
    )
    assert urls == []
    assert nested == ["https://x.test/s.xml"]
    urls, nested = parse_sitemap(
        b'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://x.test/a</loc></url></urlset>'
    )
    assert urls == ["https://x.test/a"]
    assert nested == []


def test_content_hash_is_stable():
    assert content_hash("abc") == content_hash("abc")


def test_repair_mojibake_for_nepali_utf8_as_latin1():
    broken = "नेपाल".encode("utf-8").decode("latin-1")
    assert repair_mojibake(broken) == "नेपाल"
