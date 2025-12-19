import os
from dotenv import load_dotenv
load_dotenv(verbose=True)
import time
import enum
import json
import re
from typing import TypeVar, cast
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from patchright.async_api import ElementHandle, Page
from pydantic import BaseModel
import requests
import urllib.parse

import asyncio
from browser_use import Agent
from browser_use.agent.views import ActionModel, ActionResult
from browser_use.browser.context import BrowserContext
from browser_use import BrowserConfig, Browser
from browser_use.controller.service import Controller as BrowserUseController
from browser_use.controller.registry.service import Registry
from browser_use.controller.views import (
    ClickElementAction,
    CloseTabAction,
    DoneAction,
    DragDropAction,
    GoToUrlAction,
    InputTextAction,
    NoParamsAction,
    OpenTabAction,
    Position,
    ScrollAction,
    SearchGoogleAction,
    SendKeysAction,
    SwitchTabAction,
)
from browser_use.utils import time_execution_sync
from langchain_openai import ChatOpenAI

from src.proxy.local_proxy import PROXY_URL, proxy_env
from src.tools import Tool, ToolResult
from src.logger import logger

Context = TypeVar('Context')


class FindArchiveURLAction(BaseModel):
    url: str
    date: str

class PDFAction(BaseModel):
    action: str # download, scroll_down, scroll_up, jump
    # download
    pdf_url: str | None
    save_name: str | None
    # jump
    page_number: int | None
    # scroll
    pixels: int | None
    # research
    search_text: str | None

class VideoAction(BaseModel):
    action: str # jump
    # jump
    video_url: str | None
    time: int | None

class Controller(BrowserUseController):
    def __init__(
            self,
            exclude_actions: list[str] = [],
            output_model: type[BaseModel] | None = None,
            http_save_path: str = None,
    ):
        self.http_save_path = http_save_path
        self.registry = Registry[Context](exclude_actions)

        """注册所有默认浏览器操作"""

        if output_model is not None:
            # Create a new model that extends the output model with success parameter
            class ExtendedOutputModel(BaseModel):  # type: ignore
                success: bool = True
                data: output_model  # type: ignore

            @self.registry.action(
                'Complete task - with return text and if the task is finished (success=True) or not yet  completely finished (success=False), because last step is reached',
                param_model=ExtendedOutputModel,
            )
            async def done(params: ExtendedOutputModel):
                # Exclude success from the output JSON since it's an internal parameter
                output_dict = params.data.model_dump()

                # Enums are not serializable, convert to string
                for key, value in output_dict.items():
                    if isinstance(value, enum.Enum):
                        output_dict[key] = value.value

                return ActionResult(is_done=True, success=params.success, extracted_content=json.dumps(output_dict))
        else:

            @self.registry.action(
                'Complete task - with return text and if the task is finished (success=True) or not yet  completely finished (success=False), because last step is reached',
                param_model=DoneAction,
            )
            async def done(params: DoneAction):
                return ActionResult(is_done=True, success=params.success, extracted_content=params.text)

        # Basic Navigation Actions
        @self.registry.action(
            'Search the query in Google in the current tab, the query should be a research query like humans research in Google, concrete and not vague or super long. More the single most important items. ',
            param_model=SearchGoogleAction,
        )
        async def search_google(params: SearchGoogleAction, browser: BrowserContext):
            page = await browser.get_current_page()

            encoded_query = urllib.parse.quote(params.query)

            await page.goto(f'https://www.google.com/search?q={encoded_query}&udm=14')
            await page.wait_for_load_state()
            msg = f'🔍  Searched for "{params.query}" in Google'
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action('Navigate to URL in the current tab', param_model=GoToUrlAction)
        async def go_to_url(params: GoToUrlAction, browser: BrowserContext):
            page = await browser.get_current_page()
            await page.goto(params.url)
            await page.wait_for_load_state()
            msg = f'🔗  Navigated to {params.url}'
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action('Wayback Machine for finding the archive URL of a given URL, with a specified date', param_model=FindArchiveURLAction)
        async def find_archive_url(params: FindArchiveURLAction, browser: BrowserContext):
            no_timestamp_url = f"https://archive.org/wayback/available?url={params.url}"
            archive_url = no_timestamp_url + f"&timestamp={params.date}"

            response = requests.get(archive_url).json()
            response_notimestamp = requests.get(no_timestamp_url).json()

            if "archived_snapshots" in response and "closest" in response["archived_snapshots"]:
                closest = response["archived_snapshots"]["closest"]
                logger.info(f"找到存档! {closest}")

            elif "archived_snapshots" in response_notimestamp and "closest" in response_notimestamp[
                "archived_snapshots"]:
                closest = response_notimestamp["archived_snapshots"]["closest"]
                logger.info(f"找到存档! {closest}")
            else:
                return ActionResult(
                    extracted_content = "❌  未找到给定 URL 和日期的存档 URL。",
                    include_in_memory = True,
                )

            target_url = closest["url"]
            return ActionResult(
                extracted_content = f"🕰️  Found archive URL: {target_url}",
                include_in_memory = True,
            )

        @self.registry.action('Go back', param_model=NoParamsAction)
        async def go_back(_: NoParamsAction, browser: BrowserContext):
            await browser.go_back()
            msg = '🔙  Navigated back'
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        # wait for x seconds
        @self.registry.action('Wait for x seconds default 3')
        async def wait(seconds: int = 3):
            msg = f'🕒  Waiting for {seconds} seconds'
            logger.info(msg)
            await asyncio.sleep(seconds)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        # Element Interaction Actions
        @self.registry.action('Click element by index', param_model=ClickElementAction)
        async def click_element_by_index(params: ClickElementAction, browser: BrowserContext):
            session = await browser.get_session()

            if params.index not in await browser.get_selector_map():
                raise Exception(f'Element with index {params.index} does not exist - retry or use alternative actions')

            element_node = await browser.get_dom_element_by_index(params.index)
            initial_pages = len(session.context.pages)

            # if element has file uploader then dont click
            if await browser.is_file_uploader(element_node):
                msg = f'Index {params.index} - has an element which opens file upload dialog. To upload files please use a specific function to upload files '
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

            msg = None

            try:
                download_path = await browser._click_element_node(element_node)
                if download_path:
                    msg = f'💾  Downloaded file to {download_path}'
                else:
                    msg = f'🖱️  Clicked button with index {params.index}: {element_node.get_all_text_till_next_clickable_element(max_depth=2)}'

                logger.info(msg)
                logger.debug(f'Element xpath: {element_node.xpath}')
                if len(session.context.pages) > initial_pages:
                    new_tab_msg = 'New tab opened - switching to it'
                    msg += f' - {new_tab_msg}'
                    logger.info(new_tab_msg)
                    await browser.switch_to_tab(-1)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                logger.warning(f'Element not clickable with index {params.index} - most likely the page changed')
                return ActionResult(error=str(e))

        @self.registry.action(
            'Input text into a input interactive element',
            param_model=InputTextAction,
        )
        async def input_text(params: InputTextAction, browser: BrowserContext, has_sensitive_data: bool = False):
            if params.index not in await browser.get_selector_map():
                raise Exception(f'Element index {params.index} does not exist - retry or use alternative actions')

            element_node = await browser.get_dom_element_by_index(params.index)
            await browser._input_text_element_node(element_node, params.text)
            if not has_sensitive_data:
                msg = f'⌨️  Input {params.text} into index {params.index}'
            else:
                msg = f'⌨️  Input sensitive data into index {params.index}'
            logger.info(msg)
            logger.debug(f'Element xpath: {element_node.xpath}')
            return ActionResult(extracted_content=msg, include_in_memory=True)

        # Save PDF
        @self.registry.action(
            'Save the current page as a PDF file',
        )
        async def save_pdf(browser: BrowserContext):
            page = await browser.get_current_page()
            short_url = re.sub(r'^https?://(?:www\.)?|/$', '', page.url)
            slug = re.sub(r'[^a-zA-Z0-9]+', '-', short_url).strip('-').lower()
            sanitized_filename = f'{slug}.pdf'

            await page.emulate_media(media='screen')
            await page.pdf(path=sanitized_filename, format='A4', print_background=False)
            msg = f'Saving page with URL {page.url} as PDF to ./{sanitized_filename}'
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            """在浏览器中与 Youtube 视频交互。
* `action`: 操作必须是以下之一: jump。`action` 是必需的。
- jump: 跳转到视频中的特定时间。`video_url` 和 `time` 是必需的。`video_url` 是视频的 URL，`time` 是要跳转到的秒数。
""",
            param_model=VideoAction,
        )
        async def video_viewer(params: VideoAction, browser: BrowserContext):
            page = await browser.get_current_page()

            action = params.action

            if action == "jump":

                if getattr(params, 'video_url', None) is None:
                    return ActionResult(
                        error="❌  Video URL is required for jump action.",
                        include_in_memory=True,
                    )

                if getattr(params, 'time', None) is None:
                    return ActionResult(
                        error="❌  Time is required for jump action.",
                        include_in_memory=True,
                    )

                video_url = params.video_url
                time = params.time

                jumped_url = f"{video_url}&t={time}s"

                await page.goto(jumped_url)
                await page.wait_for_load_state("networkidle")
                await page.evaluate("document.querySelector('video').play()")
                await page.wait_for_timeout(100)
                await page.evaluate("document.querySelector('video').pause()")
                await page.wait_for_timeout(100)

                current_time = await page.evaluate("document.querySelector('video').currentTime")
                await page.wait_for_timeout(1000)

                # def extract_video_id(input_str):
                #     if re.fullmatch(r"[0-9A-Za-z_-]{11}", input_str):
                #         return input_str
                #     match = re.research(r"(?:v=|\/)([0-9A-Za-z_-]{11})", input_str)
                #     return match.group(1) if match else None

                # video_id = extract_video_id(video_url)
                #
                # local_video_server_url = f"http://localhost:8080/video_viewer/player.html?v={video_id}"
                #
                # await page.goto(local_video_server_url)
                # await page.wait_for_load_state("networkidle")
                # await page.pause()
                #
                # # find the iframe with player.html
                # player_frame = next(f for f in page.frames if local_video_server_url == f.url)
                # print(player_frame.name, player_frame.url)
                #
                # await player_frame.wait_for_function("window.ytCtl && window.playerReady === true")
                # await asyncio.sleep(1)
                #
                # # await player_frame.evaluate("() => window.ytCtl.play()")
                #
                # await player_frame.evaluate(f"() => window.ytCtl.seek({time})")
                #
                # current_time = await player_frame.evaluate("() => window.ytCtl.now()")

                msg = f"🎥  Jumped to {time} seconds in the video with URL {video_url}. Current time is {current_time} seconds."
                logger.info(msg)

                return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            """从给定 URL 下载 PDF 并在浏览器中与 PDF 交互。PDF 必须先下载并在浏览器中打开，然后才能滚动或跳转到页面。
* `action`: 操作必须是以下之一: download, scroll_down, scroll_up, jump 或 research。`action` 是必需的。
 - download: 从给定 URL 下载 PDF 并在浏览器中打开。`pdf_url` 和 `save_name` 是必需的。
 - scroll_down: 将 PDF 向下滚动一页。`pixels` 是可选的。如果指定了 `pixels`，则向下滚动那么多像素。
 - scroll_up: 将 PDF 向上滚动一页。`pixels` 是可选的。如果指定了 `pixels`，则向上滚动那么多像素。
 - jump: 跳转到 PDF 中的特定页面。`page_number` 是必需的。
 - research: 在 PDF 中搜索特定文本。`search_text` 是必需的，必须是单词或短语。将滚动到文本的第一次出现。
* 当您遇到"无法加载 PDF 文档"错误时。您应该找到 pdf url（例如，https://xxx/sample.pdf），然后下载它并在浏览器中打开。
 """,
            param_model=PDFAction,
        )
        async def pdf_viewer(params: PDFAction, browser: BrowserContext):

            page = await browser.get_current_page()

            async def scroll_to_page(page, page_number):
                await page.fill('#pageNumber', str(page_number))
                await page.keyboard.press('Enter')
                await asyncio.sleep(0.5)  # Wait for the page to load

            async def scroll_to_next_page(page):
                await page.click('#next')
                await asyncio.sleep(0.5)  # Wait for the page to load

            async def scroll_to_previous_page(page):
                await page.click('#previous')
                await asyncio.sleep(0.5)  # Wait for the page to load

            action = params.action

            if action == "download":

                if not getattr(params, 'pdf_url', None):
                    return ActionResult(
                        error="❌  PDF URL is required for download action.",
                        include_in_memory=True,
                    )

                pdf_url = params.pdf_url
                save_name = params.save_name if params.save_name is not None else "downloaded_file.pdf"

                save_path = os.path.join(self.http_save_path, save_name)

                # wget command to download the PDF
                wget_command = f"wget {pdf_url} -O {save_path}"
                os.system(wget_command)

                local_pdf_server_url = f"http://localhost:8080/pdf_viewer/viewer.html?file=../local/{save_name}"

                await page.goto(local_pdf_server_url)
                await page.wait_for_selector("#viewer")
                await asyncio.sleep(5)

                msg = f"📄  Downloaded PDF from {pdf_url} to {save_path}. And opened it in browser with URL {local_pdf_server_url}."
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

            elif action == "scroll_down":

                if getattr(params, 'pixels', None):
                    pixels = params.pixels

                    await page.evaluate(f"""
                        document.querySelector('#viewerContainer').scrollTop += {pixels}
                    """)

                    await page.wait_for_load_state()

                    msg = f"📄  Scrolled down the PDF by {pixels} pixels."
                    logger.info(msg)

                    return ActionResult(extracted_content=msg, include_in_memory=True)
                else:
                    await scroll_to_next_page(page)
                    await page.wait_for_load_state()

                    msg = "📄  Scrolled down the PDF by one page."
                    logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True)
            elif action == "scroll_up":

                if getattr(params, 'pixels', None):
                    pixels = params.pixels

                    await page.evaluate(f"""
                        document.querySelector('#viewerContainer').scrollTop -= {pixels}
                    """)
                    await page.wait_for_load_state()

                    msg = f"📄  Scrolled up the PDF by {pixels} pixels."
                    logger.info(msg)

                    return ActionResult(extracted_content=msg, include_in_memory=True)
                else:
                    await scroll_to_previous_page(page)
                    await page.wait_for_load_state()

                    msg = "📄  Scrolled up the PDF by one page."
                    logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True)

            elif action == "jump":
                if getattr(params, 'page_number', None) is None:
                    return ActionResult(
                        error="❌  Page number is required for jump action.",
                        include_in_memory=True,
                    )

                page_number = params.page_number
                await scroll_to_page(page, page_number)
                await page.wait_for_load_state()

                msg = f"📄  Jumped to page {page_number} in the PDF."
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

            elif action == "research":
                if getattr(params, 'search_text', None) is None:
                    return ActionResult(
                        error="❌  Search text is required for research action.",
                        include_in_memory=True,
                    )

                search_text = params.search_text

                await page.click("#viewFind")
                await page.fill("#findInput", search_text)
                await page.keyboard.press("Enter")
                await page.wait_for_load_state()

                count_text = await page.inner_text("#findResultsCount")
                matches = re.search(r'of (\d+)', count_text)

                if not matches:
                    msg = "No matches found for the research text."
                    return ActionResult(
                        extracted_content=msg,
                        include_in_memory=True,
                    )
                else:
                    count = matches.group(1)

                    # Scroll to the first match
                    await page.click("#findNext")
                    await page.wait_for_load_state()

                    msg = f"📄  Found {count} matches for the research text '{search_text}'. Scrolled to the first match."
                    logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True)

        # Tab Management Actions
        @self.registry.action('Switch tab', param_model=SwitchTabAction)
        async def switch_tab(params: SwitchTabAction, browser: BrowserContext):
            await browser.switch_to_tab(params.page_id)
            # Wait for tab to be ready
            page = await browser.get_current_page()
            await page.wait_for_load_state()
            msg = f'🔄  Switched to tab {params.page_id}'
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action('Open url in new tab', param_model=OpenTabAction)
        async def open_tab(params: OpenTabAction, browser: BrowserContext):
            await browser.create_new_tab(params.url)
            msg = f'🔗  Opened new tab with {params.url}'
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action('Close an existing tab', param_model=CloseTabAction)
        async def close_tab(params: CloseTabAction, browser: BrowserContext):
            await browser.switch_to_tab(params.page_id)
            page = await browser.get_current_page()
            url = page.url
            await page.close()
            msg = f'❌  Closed tab #{params.page_id} with url {url}'
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        # Content Actions
        @self.registry.action(
            'Extract page content to retrieve specific information from the page, e.g. all company names, a specific description, all information about, links with companies in structured format or simply links',
        )
        async def extract_content(
                goal: str, should_strip_link_urls: bool, browser: BrowserContext, page_extraction_llm: BaseChatModel
        ):
            page = await browser.get_current_page()
            import markdownify

            strip = []
            if should_strip_link_urls:
                strip = ['a', 'img']

            content = markdownify.markdownify(await page.content(), strip=strip)

            # manually append iframe text into the content so it's readable by the LLM (includes cross-origin iframes)
            for iframe in page.frames:
                if iframe.url != page.url and not iframe.url.startswith('data:'):
                    content += f'\n\nIFRAME {iframe.url}:\n'
                    content += markdownify.markdownify(await iframe.content())

            prompt = 'Your task is to extract the content of the page. You will be given a page and a goal and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format. Extraction goal: {goal}, Page: {page}'
            template = PromptTemplate(input_variables=['goal', 'page'], template=prompt)
            try:
                output = await page_extraction_llm.ainvoke(template.format(goal=goal, page=content))
                msg = f'📄  Extracted from page\n: {output.content}\n'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                logger.debug(f'Error extracting content: {e}')
                msg = f'📄  Extracted from page\n: {content}\n'
                logger.info(msg)
                return ActionResult(extracted_content=msg)

        @self.registry.action(
            'Scroll down the page by pixel amount - if no amount is specified, scroll down one page',
            param_model=ScrollAction,
        )
        async def scroll_down(params: ScrollAction, browser: BrowserContext):
            page = await browser.get_current_page()
            if params.amount is not None:
                await page.evaluate(f'window.scrollBy(0, {params.amount});')
            else:
                await page.evaluate('window.scrollBy(0, window.innerHeight);')

            amount = f'{params.amount} pixels' if params.amount is not None else 'one page'
            msg = f'🔍  Scrolled down the page by {amount}'
            logger.info(msg)
            return ActionResult(
                extracted_content=msg,
                include_in_memory=True,
            )

        # scroll up
        @self.registry.action(
            'Scroll up the page by pixel amount - if no amount is specified, scroll up one page',
            param_model=ScrollAction,
        )
        async def scroll_up(params: ScrollAction, browser: BrowserContext):
            page = await browser.get_current_page()
            if params.amount is not None:
                await page.evaluate(f'window.scrollBy(0, -{params.amount});')
            else:
                await page.evaluate('window.scrollBy(0, -window.innerHeight);')

            amount = f'{params.amount} pixels' if params.amount is not None else 'one page'
            msg = f'🔍  Scrolled up the page by {amount}'
            logger.info(msg)
            return ActionResult(
                extracted_content=msg,
                include_in_memory=True,
            )

        # send keys
        @self.registry.action(
            'Send strings of special keys like Escape,Backspace, Insert, PageDown, Delete, Enter, Shortcuts such as `Control+o`, `Control+Shift+T` are supported as well. This gets used in keyboard.press. ',
            param_model=SendKeysAction,
        )
        async def send_keys(params: SendKeysAction, browser: BrowserContext):
            page = await browser.get_current_page()

            try:
                await page.keyboard.press(params.keys)
            except Exception as e:
                if 'Unknown key' in str(e):
                    # 遍历键并尝试发送每个键
                    for key in params.keys:
                        try:
                            await page.keyboard.press(key)
                        except Exception as e:
                            logger.debug(f'发送键 {key} 时出错: {str(e)}')
                            raise e
                else:
                    raise e
            msg = f'⌨️  Sent keys: {params.keys}'
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            description='如果您找不到想要交互的内容，请滚动到它',
        )
        async def scroll_to_text(text: str, browser: BrowserContext):  # type: ignore
            page = await browser.get_current_page()
            try:
                # Try different locator strategies
                locators = [
                    page.get_by_text(text, exact=False),
                    page.locator(f'text={text}'),
                    page.locator(f"//*[contains(text(), '{text}')]"),
                ]

                for locator in locators:
                    try:
                        # First check if element exists and is visible
                        if await locator.count() > 0 and await locator.first.is_visible():
                            await locator.first.scroll_into_view_if_needed()
                            await asyncio.sleep(0.5)  # Wait for scroll to complete
                            msg = f'🔍  Scrolled to text: {text}'
                            logger.info(msg)
                            return ActionResult(extracted_content=msg, include_in_memory=True)
                    except Exception as e:
                        logger.debug(f'定位器尝试失败: {str(e)}')
                        continue

                msg = f"Text '{text}' not found or not visible on page"
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

            except Exception as e:
                msg = f"滚动到文本 '{text}' 失败: {str(e)}"
                logger.error(msg)
                return ActionResult(error=msg, include_in_memory=True)

        @self.registry.action(
            description='从原生下拉菜单获取所有选项',
        )
        async def get_dropdown_options(index: int, browser: BrowserContext) -> ActionResult:
            """从原生下拉菜单获取所有选项"""
            page = await browser.get_current_page()
            selector_map = await browser.get_selector_map()
            dom_element = selector_map[index]

            try:
                # Frame-aware approach since we know it works
                all_options = []
                frame_index = 0

                for frame in page.frames:
                    try:
                        options = await frame.evaluate(
                            """
                            (xpath) => {
                                const select = document.evaluate(xpath, document, null,
                                    XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                if (!select) return null;

                                return {
                                    options: Array.from(select.options).map(opt => ({
                                        text: opt.text, //do not trim, because we are doing exact match in select_dropdown_option
                                        value: opt.value,
                                        index: opt.index
                                    })),
                                    id: select.id,
                                    name: select.name
                                };
                            }
                        """,
                            dom_element.xpath,
                        )

                        if options:
                            logger.debug(f'Found dropdown in frame {frame_index}')
                            logger.debug(f'Dropdown ID: {options["id"]}, Name: {options["name"]}')

                            formatted_options = []
                            for opt in options['options']:
                                # encoding ensures AI uses the exact string in select_dropdown_option
                                encoded_text = json.dumps(opt['text'])
                                formatted_options.append(f'{opt["index"]}: text={encoded_text}')

                            all_options.extend(formatted_options)

                    except Exception as frame_e:
                        logger.debug(f'框架 {frame_index} 评估失败: {str(frame_e)}')

                    frame_index += 1

                if all_options:
                    msg = '\n'.join(all_options)
                    msg += '\nUse the exact text string in select_dropdown_option'
                    logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True)
                else:
                    msg = 'No options found in any frame for dropdown'
                    logger.info(msg)
                    return ActionResult(extracted_content=msg, include_in_memory=True)

            except Exception as e:
                logger.error(f'获取下拉选项失败: {str(e)}')
                msg = f'获取选项时出错: {str(e)}'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            description='通过要选择的选项文本来选择交互元素索引的下拉选项',
        )
        async def select_dropdown_option(
                index: int,
                text: str,
                browser: BrowserContext,
        ) -> ActionResult:
            """通过要选择的选项文本来选择下拉选项"""
            page = await browser.get_current_page()
            selector_map = await browser.get_selector_map()
            dom_element = selector_map[index]

            # Validate that we're working with a select element
            if dom_element.tag_name != 'select':
                logger.error(f'Element is not a select! Tag: {dom_element.tag_name}, Attributes: {dom_element.attributes}')
                msg = f'Cannot select option: Element with index {index} is a {dom_element.tag_name}, not a select'
                return ActionResult(extracted_content=msg, include_in_memory=True)

            logger.debug(f"Attempting to select '{text}' using xpath: {dom_element.xpath}")
            logger.debug(f'Element attributes: {dom_element.attributes}')
            logger.debug(f'Element tag: {dom_element.tag_name}')

            xpath = '//' + dom_element.xpath

            try:
                frame_index = 0
                for frame in page.frames:
                    try:
                        logger.debug(f'Trying frame {frame_index} URL: {frame.url}')

                        # First verify we can find the dropdown in this frame
                        find_dropdown_js = """
							(xpath) => {
								try {
									const select = document.evaluate(xpath, document, null,
										XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
									if (!select) return null;
									if (select.tagName.toLowerCase() !== 'select') {
										return {
											error: `找到元素但它是 ${select.tagName}，不是 SELECT`,
											found: false
										};
									}
									return {
										id: select.id,
										name: select.name,
										found: true,
										tagName: select.tagName,
										optionCount: select.options.length,
										currentValue: select.value,
										availableOptions: Array.from(select.options).map(o => o.text.trim())
									};
								} catch (e) {
									return {error: e.toString(), found: false};
								}
							}
						"""

                        dropdown_info = await frame.evaluate(find_dropdown_js, dom_element.xpath)

                        if dropdown_info:
                            if not dropdown_info.get('found'):
                                logger.error(f'Frame {frame_index} error: {dropdown_info.get("error")}')
                                continue

                            logger.debug(f'Found dropdown in frame {frame_index}: {dropdown_info}')

                            # "label" because we are selecting by text
                            # nth(0) to disable error thrown by strict mode
                            # timeout=1000 because we are already waiting for all network events, therefore ideally we don't need to wait a lot here (default 30s)
                            selected_option_values = (
                                await frame.locator('//' + dom_element.xpath).nth(0).select_option(label=text, timeout=1000)
                            )

                            msg = f'selected option {text} with value {selected_option_values}'
                            logger.info(msg + f' in frame {frame_index}')

                            return ActionResult(extracted_content=msg, include_in_memory=True)

                    except Exception as frame_e:
                        logger.error(f'Frame {frame_index} attempt failed: {str(frame_e)}')
                        logger.error(f'Frame type: {type(frame)}')
                        logger.error(f'Frame URL: {frame.url}')

                    frame_index += 1

                msg = f"Could not select option '{text}' in any frame"
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

            except Exception as e:
                msg = f'Selection failed: {str(e)}'
                logger.error(msg)
                return ActionResult(error=msg, include_in_memory=True)

        @self.registry.action(
            'Drag and drop elements or between coordinates on the page - useful for canvas drawing, sortable lists, sliders, file uploads, and UI rearrangement',
            param_model=DragDropAction,
        )
        async def drag_drop(params: DragDropAction, browser: BrowserContext) -> ActionResult:
            """
            Performs a precise drag and drop operation between elements or coordinates.
            """

            async def get_drag_elements(
                    page: Page,
                    source_selector: str,
                    target_selector: str,
            ) -> tuple[ElementHandle | None, ElementHandle | None]:
                """Get source and target elements with appropriate error handling."""
                source_element = None
                target_element = None

                try:
                    # page.locator() auto-detects CSS and XPath
                    source_locator = page.locator(source_selector)
                    target_locator = page.locator(target_selector)

                    # Check if elements exist
                    source_count = await source_locator.count()
                    target_count = await target_locator.count()

                    if source_count > 0:
                        source_element = await source_locator.first.element_handle()
                        logger.debug(f'Found source element with selector: {source_selector}')
                    else:
                        logger.warning(f'Source element not found: {source_selector}')

                    if target_count > 0:
                        target_element = await target_locator.first.element_handle()
                        logger.debug(f'Found target element with selector: {target_selector}')
                    else:
                        logger.warning(f'Target element not found: {target_selector}')

                except Exception as e:
                    logger.error(f'Error finding elements: {str(e)}')

                return source_element, target_element

            async def get_element_coordinates(
                    source_element: ElementHandle,
                    target_element: ElementHandle,
                    source_position: Position | None,
                    target_position: Position | None,
            ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
                """Get coordinates from elements with appropriate error handling."""
                source_coords = None
                target_coords = None

                try:
                    # Get source coordinates
                    if source_position:
                        source_coords = (source_position.x, source_position.y)
                    else:
                        source_box = await source_element.bounding_box()
                        if source_box:
                            source_coords = (
                                int(source_box['x'] + source_box['width'] / 2),
                                int(source_box['y'] + source_box['height'] / 2),
                            )

                    # Get target coordinates
                    if target_position:
                        target_coords = (target_position.x, target_position.y)
                    else:
                        target_box = await target_element.bounding_box()
                        if target_box:
                            target_coords = (
                                int(target_box['x'] + target_box['width'] / 2),
                                int(target_box['y'] + target_box['height'] / 2),
                            )
                except Exception as e:
                    logger.error(f'Error getting element coordinates: {str(e)}')

                return source_coords, target_coords

            async def execute_drag_operation(
                    page: Page,
                    source_x: int,
                    source_y: int,
                    target_x: int,
                    target_y: int,
                    steps: int,
                    delay_ms: int,
            ) -> tuple[bool, str]:
                """Execute the drag operation with comprehensive error handling."""
                try:
                    # Try to move to source position
                    try:
                        await page.mouse.move(source_x, source_y)
                        logger.debug(f'Moved to source position ({source_x}, {source_y})')
                    except Exception as e:
                        logger.error(f'移动到源位置失败: {str(e)}')
                        return False, f'移动到源位置失败: {str(e)}'

                    # Press mouse button down
                    await page.mouse.down()

                    # Move to target position with intermediate steps
                    for i in range(1, steps + 1):
                        ratio = i / steps
                        intermediate_x = int(source_x + (target_x - source_x) * ratio)
                        intermediate_y = int(source_y + (target_y - source_y) * ratio)

                        await page.mouse.move(intermediate_x, intermediate_y)

                        if delay_ms > 0:
                            await asyncio.sleep(delay_ms / 1000)

                    # Move to final target position
                    await page.mouse.move(target_x, target_y)

                    # Move again to ensure dragover events are properly triggered
                    await page.mouse.move(target_x, target_y)

                    # Release mouse button
                    await page.mouse.up()

                    return True, 'Drag operation completed successfully'

                except Exception as e:
                    return False, f'Error during drag operation: {str(e)}'

            page = await browser.get_current_page()

            try:
                # Initialize variables
                source_x: int | None = None
                source_y: int | None = None
                target_x: int | None = None
                target_y: int | None = None

                # Normalize parameters
                steps = max(1, params.steps or 10)
                delay_ms = max(0, params.delay_ms or 5)

                # Case 1: Element selectors provided
                if params.element_source and params.element_target:
                    logger.debug('Using element-based approach with selectors')

                    source_element, target_element = await get_drag_elements(
                        page,
                        params.element_source,
                        params.element_target,
                    )

                    if not source_element or not target_element:
                        error_msg = f'找不到{"源" if not source_element else "目标"}元素'
                        return ActionResult(error=error_msg, include_in_memory=True)

                    source_coords, target_coords = await get_element_coordinates(
                        source_element, target_element, params.element_source_offset, params.element_target_offset
                    )

                    if not source_coords or not target_coords:
                        error_msg = f'无法确定{"源" if not source_coords else "目标"}坐标'
                        return ActionResult(error=error_msg, include_in_memory=True)

                    source_x, source_y = source_coords
                    target_x, target_y = target_coords

                # Case 2: Coordinates provided directly
                elif all(
                        coord is not None
                        for coord in [params.coord_source_x, params.coord_source_y, params.coord_target_x, params.coord_target_y]
                ):
                    logger.debug('Using coordinate-based approach')
                    source_x = params.coord_source_x
                    source_y = params.coord_source_y
                    target_x = params.coord_target_x
                    target_y = params.coord_target_y
                else:
                    error_msg = '必须提供源/目标选择器或源/目标坐标'
                    return ActionResult(error=error_msg, include_in_memory=True)

                # Validate coordinates
                if any(coord is None for coord in [source_x, source_y, target_x, target_y]):
                    error_msg = 'Failed to determine source or target coordinates'
                    return ActionResult(error=error_msg, include_in_memory=True)

                # Perform the drag operation
                success, message = await execute_drag_operation(
                    page,
                    cast(int, source_x),
                    cast(int, source_y),
                    cast(int, target_x),
                    cast(int, target_y),
                    steps,
                    delay_ms,
                )

                if not success:
                    logger.error(f'Drag operation failed: {message}')
                    return ActionResult(error=message, include_in_memory=True)

                # Create descriptive message
                if params.element_source and params.element_target:
                    msg = f"🖱️ Dragged element '{params.element_source}' to '{params.element_target}'"
                else:
                    msg = f'🖱️ Dragged from ({source_x}, {source_y}) to ({target_x}, {target_y})'

                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)

            except Exception as e:
                error_msg = f'执行拖放操作失败: {str(e)}'
                logger.error(error_msg)
                return ActionResult(error=error_msg, include_in_memory=True)

        @self.registry.action('Google Sheets: Get the contents of the entire sheet', domains=['sheets.google.com'])
        async def get_sheet_contents(browser: BrowserContext):
            page = await browser.get_current_page()

            # select all cells
            await page.keyboard.press('Enter')
            await page.keyboard.press('Escape')
            await page.keyboard.press('ControlOrMeta+A')
            await page.keyboard.press('ControlOrMeta+C')

            extracted_tsv = await page.evaluate('() => navigator.clipboard.readText()')
            return ActionResult(extracted_content=extracted_tsv, include_in_memory=True)

        @self.registry.action('Google Sheets: Select a specific cell or range of cells', domains=['sheets.google.com'])
        async def select_cell_or_range(browser: BrowserContext, cell_or_range: str):
            page = await browser.get_current_page()

            await page.keyboard.press('Enter')  # make sure we dont delete current cell contents if we were last editing
            await page.keyboard.press('Escape')  # to clear current focus (otherwise select range popup is additive)
            await asyncio.sleep(0.1)
            await page.keyboard.press('Home')  # move cursor to the top left of the sheet first
            await page.keyboard.press('ArrowUp')
            await asyncio.sleep(0.1)
            await page.keyboard.press('Control+G')  # open the goto range popup
            await asyncio.sleep(0.2)
            await page.keyboard.type(cell_or_range, delay=0.05)
            await asyncio.sleep(0.2)
            await page.keyboard.press('Enter')
            await asyncio.sleep(0.2)
            await page.keyboard.press('Escape')  # to make sure the popup still closes in the case where the jump failed
            return ActionResult(extracted_content=f'Selected cell {cell_or_range}', include_in_memory=False)

        @self.registry.action(
            'Google Sheets: Get the contents of a specific cell or range of cells', domains=['sheets.google.com']
        )
        async def get_range_contents(browser: BrowserContext, cell_or_range: str):
            page = await browser.get_current_page()

            await select_cell_or_range(browser, cell_or_range)

            await page.keyboard.press('ControlOrMeta+C')
            await asyncio.sleep(0.1)
            extracted_tsv = await page.evaluate('() => navigator.clipboard.readText()')
            return ActionResult(extracted_content=extracted_tsv, include_in_memory=True)

        @self.registry.action('Google Sheets: Clear the currently selected cells', domains=['sheets.google.com'])
        async def clear_selected_range(browser: BrowserContext):
            page = await browser.get_current_page()

            await page.keyboard.press('Backspace')
            return ActionResult(extracted_content='Cleared selected range', include_in_memory=False)

        @self.registry.action('Google Sheets: Input text into the currently selected cell', domains=['sheets.google.com'])
        async def input_selected_cell_text(browser: BrowserContext, text: str):
            page = await browser.get_current_page()

            await page.keyboard.type(text, delay=0.1)
            await page.keyboard.press('Enter')  # make sure to commit the input so it doesn't get overwritten by the next action
            await page.keyboard.press('ArrowUp')
            return ActionResult(extracted_content=f'Inputted text {text}', include_in_memory=False)

        @self.registry.action('Google Sheets: Batch update a range of cells', domains=['sheets.google.com'])
        async def update_range_contents(browser: BrowserContext, range: str, new_contents_tsv: str):
            page = await browser.get_current_page()

            await select_cell_or_range(browser, range)

            # simulate paste event from clipboard with TSV content
            await page.evaluate(f"""
				const clipboardData = new DataTransfer();
				clipboardData.setData('text/plain', `{new_contents_tsv}`);
				document.activeElement.dispatchEvent(new ClipboardEvent('paste', {{clipboardData}}));
			""")

            return ActionResult(extracted_content=f'Updated cell {range} with {new_contents_tsv}', include_in_memory=False)

    # Register ---------------------------------------------------------------

    def action(self, description: str, **kwargs):
        """Decorator for registering custom actions

        @param description: Describe the LLM what the function does (better description == better function calling)
        """
        return self.registry.action(description, **kwargs)

    # Act --------------------------------------------------------------------

    @time_execution_sync('--act')
    async def act(
            self,
            action: ActionModel,
            browser_context: BrowserContext,
            #
            page_extraction_llm: BaseChatModel | None = None,
            sensitive_data: dict[str, str] | None = None,
            available_file_paths: list[str] | None = None,
            #
            context: Context | None = None,
    ) -> ActionResult:
        """Execute an action"""

        try:
            for action_name, params in action.model_dump(exclude_unset=True).items():
                if params is not None:
                    # with Laminar.start_as_current_span(
                    # 	name=action_name,
                    # 	input={
                    # 		'action': action_name,
                    # 		'params': params,
                    # 	},
                    # 	span_type='TOOL',
                    # ):
                    result = await self.registry.execute_action(
                        action_name,
                        params,
                        browser=browser_context,
                        page_extraction_llm=page_extraction_llm,
                        sensitive_data=sensitive_data,
                        available_file_paths=available_file_paths,
                        context=context,
                    )

                    # Laminar.set_span_output(result)

                    if isinstance(result, str):
                        return ActionResult(extracted_content=result)
                    elif isinstance(result, ActionResult):
                        return result
                    elif result is None:
                        return ActionResult()
                    else:
                        raise ValueError(f'Invalid action result type: {type(result)} of {result}')
            return ActionResult()
        except Exception as e:
            raise e