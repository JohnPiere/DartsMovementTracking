# Source export notes — ssn.or.jp / business4.html

## What I was asked to do
Scrape <https://www.ssn.or.jp/business4.html>, export all its data and pictures,
and store them; then build a fresh modern 3D website from it.

## Important: why the original assets are NOT in this folder
This task ran inside a **sandboxed cloud execution environment** whose outbound
network access is restricted by an **allowlist policy**. The host `ssn.or.jp`
(and also `web.archive.org` and `google.com`, plus the public CDNs jsDelivr /
unpkg / cdnjs) are **not on the allowlist**, so every direct fetch was rejected:

```
$ curl https://www.ssn.or.jp/business4.html
HTTP 403 — "Host not in allowlist"
```

That means I could **not** download the original HTML, text, or image binaries
(`.jpg/.png`) from the live site. There was no way around it from here — it is an
infrastructure restriction, not a code problem.

### How to get the real, byte-for-byte assets
Run this on your own machine at home (it has unrestricted internet):

```bash
# from inside the sendai-seika-modern/source-export/ folder
wget --mirror --convert-links --adjust-extension --page-requisites \
     --no-parent -e robots=off \
     --user-agent="Mozilla/5.0" \
     https://www.ssn.or.jp/business4.html

# or grab the whole site:
wget -mkEpnp -e robots=off https://www.ssn.or.jp/
```

That will drop the original `business4.html` plus its `/images/*` into this
folder. The modern site in `../` does not depend on those files — it uses freshly
generated 3D/CSS artwork — so you can mirror them purely for reference.

## What I reconstructed instead (via web search)
The page content below was reassembled from search-engine snippets so the
redesign is faithful in **meaning**. Treat the Japanese body copy on the new
site as *representative*, not a verbatim copy of the original page — swap in the
real text once you mirror the site.

See `content.md` for the reconstructed content map.
