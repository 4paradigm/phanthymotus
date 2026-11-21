"""Real browser QA for source-only labeling, not a human annotation session."""
import argparse
import hashlib
import json
from pathlib import Path
import threading
from annotation_workbench import create_server


def main():
    p=argparse.ArgumentParser();p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--frames',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    digest=hashlib.sha256(args.labels.read_bytes()).hexdigest()
    server=create_server(args.labels,args.frames);worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    base=f'http://127.0.0.1:{server.server_port}';errors=[]
    from playwright.sync_api import sync_playwright,expect
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1600,'height':1100},accept_downloads=True)
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.route('**/*',lambda r:r.continue_() if r.request.url.startswith((base,'blob:','data:')) else r.abort())
            page.goto(base,wait_until='domcontentloaded')
            expect(page.locator('#status')).to_contain_text('close_speaker / frame 0 / snapshot',timeout=15000)
            count=page.locator('#frame option').count();assert count==62
            page.locator('#zoom').select_option('1')
            page.locator('#reviewer').fill('UI_TEST_ONLY_DO_NOT_SCORE')
            page.locator('#point').select_option('2')
            box=page.locator('#canvas').bounding_box()
            page.mouse.click(box['x']+500*box['width']/1280,box['y']+300*box['height']/720)
            expect(page.locator('#points')).to_contain_text('2: 500, 300')
            page.locator('#hidden').click();expect(page.locator('#points')).to_contain_text('2: 不可见/未标注')
            page.mouse.click(box['x']+500*box['width']/1280,box['y']+300*box['height']/720)
            page.locator('#review').click()
            nose_index=next(k for k,text in enumerate(page.locator('#frame option').all_text_contents()) if text.startswith('鼻轨迹'))
            page.locator('#frame').select_option(str(nose_index))
            expect(page.locator('#status')).to_contain_text('close_listener / frame 0 / nose')
            assert page.locator('#point').is_disabled()
            box=page.locator('#canvas').bounding_box()
            page.mouse.click(box['x']+610*box['width']/1280,box['y']+280*box['height']/720)
            expect(page.locator('#points')).to_contain_text('2: 610, 280')
            page.locator('#review').click();page.locator('#attest').check()
            with page.expect_download() as download:
                page.locator('#export').click()
            target=args.output/'ui-test-export-not-ground-truth.json';download.value.save_as(str(target))
            result=json.loads(target.read_text());assert not result['independent_human_review']
            assert result['clips']['close_speaker'][0]['faces'][0]['points'][2]==[500,300]
            assert result['nose_burst']['nose_xy'][0]==[610,280]
            assert next(f for f in result['clips']['close_listener'][0]['faces'] if f['id']==result['nose_burst']['identity'])['points'][2]==[610,280]
            assert len(result['annotation_review']['reviewed_items'])==2
            assert len(result['annotation_review']['source_images'])==61
            assert result['annotator']=='UI_TEST_ONLY_DO_NOT_SCORE'
            page.screenshot(path=str(args.output/'annotation-workbench.png'),full_page=True)
            assert not errors
            browser.close()
        assert hashlib.sha256(args.labels.read_bytes()).hexdigest()==digest
        report={'pass':True,'frame_items':count,'unique_source_images':61,'pixel_coordinate_edit_verified':True,
                'hidden_point_null_verified':True,'same_frame_nose_sync_verified':True,
                'partial_export_stays_unreviewed':True,'original_labels_unchanged':True,
                'page_errors':errors,'scope':'real browser UI QA with artificial edits; not human ground truth or platform integration'}
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    finally:server.shutdown();server.server_close();worker.join(3)


if __name__=='__main__':main()
