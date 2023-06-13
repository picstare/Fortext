
import streamlit as st
import asyncio
from playwright.async_api import async_playwright


st.write("Starting the test…")


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto("https://facebook.com/pemkabgresik")

        

        # Scroll to the bottom of the page with a limit of 10 scrolls
        scroll_limit = 10
        current_scroll = 0

        while current_scroll < scroll_limit:
            await page.evaluate('window.scrollBy(0, window.innerHeight);')
            await asyncio.sleep(0.1)  # Adjust the delay as needed
            current_scroll += 1

        # Extract posts
        title = await page.title()
        posts_locator = ".x1a2a7pz[role=article]"
        posts = await page.query_selector_all(posts_locator)

        post_texts = []
        for post in posts:
            post_text = await post.text_content()
            post_texts.append(post_text)

        # Streamlit code
        st.write(title)
        st.write(post_texts)

        await browser.close()
        return title


if __name__ == '__main__':
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    title = loop.run_until_complete(main())
    print(title)

    