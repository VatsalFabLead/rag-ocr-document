# PHP & Laravel Client SDK for FastAPI Custom OCR Pipeline

Connect your PHP / Laravel projects to the FastAPI Document Intelligence & OCR Server with zero external cloud dependencies.

---

## 🚀 Quick Usage in Laravel

### Option 1: Drop-in Class
1. Copy [CustomOcrClient.php](file:///d:/Python/RAG%20Document/custom_ocr/sdk/php/CustomOcrClient.php) into your Laravel project under `app/Services/Ocr/CustomOcrClient.php`.
2. Add your server URL and Master Key to your Laravel `.env` file:
   ```env
   CUSTOM_OCR_BASE_URL=http://127.0.0.1:8000
   CUSTOM_OCR_API_KEY=sk-custom-ocr-master-v1.1
   ```
3. Use it inside any Controller, Job, or Service:
   ```php
   use App\Services\Ocr\CustomOcrClient;

   class DocumentController extends Controller
   {
       public function process(Request $request)
       {
           $ocrClient = new CustomOcrClient(
               config('services.ocr.base_url', env('CUSTOM_OCR_BASE_URL', 'http://127.0.0.1:8000')),
               config('services.ocr.api_key', env('CUSTOM_OCR_API_KEY', 'sk-custom-ocr-master-v1.1'))
           );

           // 1. Process uploaded file
           $file = $request->file('document');
           $result = $ocrClient->predictFile($file->getRealPath());

           // Extracted entities
           $rawText = $result['raw_text'];
           $fields = $result['fields'];
           $tables = $result['tables'];

           return response()->json([
               'success' => true,
               'text' => $rawText,
               'fields' => $fields,
           ]);
       }
   }
   ```

---

### Option 2: Direct Laravel `Http` Client (No Extra Files Needed)
You can also use Laravel's native `Http` facade directly without any external library:

```php
use Illuminate\Support\Facades\Http;

$response = Http::withHeaders([
    'Authorization' => 'Bearer sk-custom-ocr-master-v1.1',
])->attach(
    'file', 
    file_get_contents($uploadedFile->getRealPath()), 
    $uploadedFile->getClientOriginalName()
)->post('http://127.0.0.1:8000/api/ocr/predict', [
    'engine' => 'auto',
    'extract_fields' => 'true',
    'extract_tables' => 'true',
]);

if ($response->successful()) {
    $data = $response->json();
    $rawText = $data['raw_text'];
    $fields = $data['fields'];
}
```

---

## 📡 Base64 OCR (For Camera captures & Scanned crops)
```php
$ocrClient = new CustomOcrClient();

$base64Image = base64_encode(file_get_contents($imagePath));
$result = $ocrClient->predictBase64($base64Image, 'receipt.jpg');

print_r($result['fields']);
```

---

## 🤖 Ask Questions on Document (RAG Grounded Q&A)
```php
$result = $ocrClient->askRagQuestion($docId, 'What is the invoice total amount?');
echo $result['answer'];
```
